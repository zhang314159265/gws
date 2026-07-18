#!/usr/bin/env python3
"""
Delete (or trash) all Gmail messages carrying a given label.

Setup:
  1. pip install google-auth-oauthlib google-api-python-client
  2. In Google Cloud Console, enable the Gmail API and create OAuth
     "Desktop app" credentials. Download the JSON as credentials.json
     and place it next to this script (or pass --credentials).
  3. First run opens a browser to authorize; a token.json is cached
     next to this script for subsequent runs.

Usage:
  # Move all messages labeled "Promotions/OldStuff" to Trash (reversible).
  python gmail_delete_by_label.py "Promotions/OldStuff"

  # Preview what would happen without changing anything.
  python gmail_delete_by_label.py "Promotions/OldStuff" --dry-run

  # Permanently delete instead of trashing (irreversible).
  python gmail_delete_by_label.py "Promotions/OldStuff" --permanent

  # Only process the first 50 matching messages (useful for testing).
  python gmail_delete_by_label.py "Promotions/OldStuff" --max-messages 50
"""
import argparse
import os

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(SCRIPT_DIR, "token.json")
DEFAULT_CREDENTIALS_PATH = os.path.join(SCRIPT_DIR, "credentials.json")

# gmail.modify covers trashing; permanent deletion needs the broader mail scope.
SCOPES = ["https://mail.google.com/"]

BATCH_DELETE_SIZE = 1000


def get_service(credentials_path):
    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def resolve_label_id(service, label_name):
    labels = service.users().labels().list(userId="me").execute().get("labels", [])
    for label in labels:
        if label["name"].lower() == label_name.lower():
            return label["id"]
    available = ", ".join(sorted(l["name"] for l in labels))
    raise SystemExit(f'Label "{label_name}" not found. Available labels: {available}')


def list_message_ids(service, label_id, max_messages=None):
    message_ids = []
    page_token = None
    while True:
        page_size = 500
        if max_messages is not None:
            page_size = min(page_size, max_messages - len(message_ids))
        response = (
            service.users()
            .messages()
            .list(
                userId="me",
                labelIds=[label_id],
                pageToken=page_token,
                maxResults=page_size,
            )
            .execute()
        )
        message_ids.extend(m["id"] for m in response.get("messages", []))
        if max_messages is not None and len(message_ids) >= max_messages:
            return message_ids[:max_messages]
        page_token = response.get("nextPageToken")
        if not page_token:
            break
    return message_ids


def trash_messages(service, message_ids):
    for start in range(0, len(message_ids), BATCH_DELETE_SIZE):
        chunk = message_ids[start : start + BATCH_DELETE_SIZE]
        service.users().messages().batchModify(
            userId="me", body={"ids": chunk, "addLabelIds": ["TRASH"]}
        ).execute()
        print(f"Trashed {start + len(chunk)}/{len(message_ids)}")


def permanently_delete_messages(service, message_ids):
    for start in range(0, len(message_ids), BATCH_DELETE_SIZE):
        chunk = message_ids[start : start + BATCH_DELETE_SIZE]
        service.users().messages().batchDelete(
            userId="me", body={"ids": chunk}
        ).execute()
        print(f"Permanently deleted {start + len(chunk)}/{len(message_ids)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("label", help="Gmail label name (case-insensitive)")
    parser.add_argument(
        "--credentials",
        default=DEFAULT_CREDENTIALS_PATH,
        help="Path to OAuth client credentials.json",
    )
    parser.add_argument(
        "--permanent",
        action="store_true",
        help="Permanently delete instead of moving to Trash (irreversible)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show how many messages match; make no changes",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt",
    )
    parser.add_argument(
        "--max-messages",
        type=int,
        default=None,
        help="Stop once this many matching messages are found (default: no cap)",
    )
    args = parser.parse_args()

    service = get_service(args.credentials)
    label_id = resolve_label_id(service, args.label)
    message_ids = list_message_ids(service, label_id, max_messages=args.max_messages)

    print(f'Found {len(message_ids)} message(s) with label "{args.label}".')
    if not message_ids:
        return
    if args.dry_run:
        return

    if not args.yes:
        action = "PERMANENTLY DELETE" if args.permanent else "move to Trash"
        confirm = input(f"{action} {len(message_ids)} message(s)? [y/N] ")
        if confirm.strip().lower() != "y":
            print("Aborted.")
            return

    if args.permanent:
        permanently_delete_messages(service, message_ids)
    else:
        trash_messages(service, message_ids)


if __name__ == "__main__":
    main()
