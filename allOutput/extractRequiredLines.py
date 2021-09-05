import sys

fname = sys.argv[1]

fptr = open(fname)

selLines = [44, 56,60,64,68,72,116,128,132,136,140,144,188,200,204,208,212,216]

linenum = 1
for line in fptr:
	
	if linenum in selLines:
		print(line.strip())
		if linenum == 72 or linenum == 144:
			print("\n")

	linenum = linenum + 1