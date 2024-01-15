find ./tmp/data/ -size 0 -exec rm {} +
find ./tmp/data/ -type f ! -name "*.jpg" -exec rm {} +