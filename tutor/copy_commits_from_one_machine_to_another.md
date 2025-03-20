From source repo:
  ```
  git format-patch -o /tmp/patch main..shunting-change
  ```

  A folder called /tmp/patch will be created with one .patch file for reach commit.

From dst repo:
  ```
  git am /tmp/patch/*.patch
  ```
