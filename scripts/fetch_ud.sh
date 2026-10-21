#!/usr/bin/env bash
# Downloads UD English EWT (CC BY-SA 4.0) into data/ud-raw/ (not committed).
set -euo pipefail
dir="$(cd "$(dirname "$0")/.." && pwd)/data/ud-raw"
mkdir -p "$dir"
for s in train dev test; do
  curl -sSfL "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-$s.conllu" -o "$dir/en_ewt-ud-$s.conllu"
done
curl -sSfL "https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/LICENSE.txt" -o "$dir/LICENSE.txt"
echo "UD EWT downloaded to $dir"
