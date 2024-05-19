#pragma once

namespace toy {

struct Location {
  std::shared_ptr<std::string> file;
  int line;
  int col;
};

enum Token : int {
  tok_eof = -1,

  // commands
  tok_return = -2,
  tok_var = -3,
  tok_def = -4,

  // primary
  tok_identifier = -5,
  tok_number = -6,
};

class Lexer {
 public:
  Lexer(std::string filename)
      : lastLocation(
          {std::make_shared<std::string>(std::move(filename)), 0, 0}) {}

  virtual ~Lexer() = default;

  Token getNextToken() {
    return curTok = getTok();
  }

  Token getCurToken() { return curTok; }

  Location getLastLocation() { return lastLocation; }

  void consume(Token tok) {
    assert(tok == curTok && "consume Token mismatch expectation");
    getNextToken();
  }

  llvm::StringRef getId() {
    assert(curTok == tok_identifier);
    return identifierStr;
  }

  double getValue() {
    assert(curTok == tok_number);
    return numVal;
  }
 private:
  virtual llvm::StringRef readNextLine() = 0;


  int getNextChar() {
    // The current line buffer should not be empty unless it is the end of file.
    if (curLineBuffer.empty()) {
      return EOF;
    }
    ++curCol;
    auto nextchar = curLineBuffer.front();
    curLineBuffer = curLineBuffer.drop_front();
    if (curLineBuffer.empty()) {
      curLineBuffer = readNextLine();
    }
    if (nextchar == '\n') {
      ++curLineNum;
      curCol = 0;
    }
    return nextchar;
  }

  // Return the next token from standard input.
  Token getTok() {
    // Skip any whitespace.
    while (isspace(lastChar)) {
      lastChar = Token(getNextChar());
    }

    // Save the current location before reading the token characters
    lastLocation.line = curLineNum;
    lastLocation.col = curCol;

    // Identifier
    if (isalpha(lastChar)) {
      identifierStr = (char) lastChar;
      while (isalnum(lastChar = Token(getNextChar())) || lastChar == '_') {
        identifierStr += (char) lastChar;
      }

      if (identifierStr == "return") {
        return tok_return;
      } else if (identifierStr == "def") {
        return tok_def;
      } else if (identifierStr == "var") {
        return tok_var;
      } else {
        return tok_identifier;
      }
    }

    // Number
    if (isdigit(lastChar) || lastChar == '.') {
      std::string numStr;
      do {
        numStr += lastChar;
        lastChar = Token(getNextChar());
      } while (isdigit(lastChar) || lastChar == '.');

      numVal = strtod(numStr.c_str(), nullptr);
      return tok_number;
    }

    if (lastChar == '#') {
      assert(false && "comment");
    }

    if (lastChar == EOF) {
      return tok_eof;
    }

    // Otherwise, just return the character as its ascii value.
    Token thisChar = Token(lastChar);
    lastChar = Token(getNextChar());
    return thisChar;
  }

  // The last token read from the input
  Token curTok = tok_eof;

  std::string identifierStr;
  double numVal = 0;

  Token lastChar = Token(' ');

  Location lastLocation;

  int curLineNum = 0;
  int curCol = 0;
  llvm::StringRef curLineBuffer = "\n";
};

class LexerBuffer final : public Lexer {
 public:
  LexerBuffer(const char *begin, const char *end, std::string filename)
      : Lexer(std::move(filename)), current(begin), end(end) { }

 private:
  llvm::StringRef readNextLine() override {
    auto *begin = current;
    while (current <= end && *current && *current != '\n') {
      ++current;
    }
    if (current <= end && *current) {
      ++current;
    }
    llvm::StringRef result{begin, static_cast<size_t>(current - begin)};
    return result;
  }

  const char *current, *end;
};

}
