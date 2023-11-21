ps aux | awk '/[b]oard.py|[t]wod.py/ {print $2}'  | xargs kill
