#!/bin/bash

# The script assumes GEMINI_API_KEY in the environment.

curl https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$GEMINI_API_KEY -H "Content-Type: application/json" -d '
{
  "contents": [{
    "parts": [{
      "text": "Explain quicksort in one sentense."
    }]
  }]
}
'
