/<Proper / {
	max = 0
	for (i = 1; i <= NF; i++) {
		matched = match($i, /phase([1-6])/, arr)
		if (matched && arr[1] > max) max = arr[1]
	}
	for (i = max; i > 0; i--) {
		$1 = $1 sprintf(" idivf%d=\"1\"", i)
	}
	$1 = $1 " "
	print
	next
}

{ print }
