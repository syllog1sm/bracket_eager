import sys
for line in sys.stdin:
   line = line.strip()
   assert line[:5] == "(TOP "
   assert line[-2:] == " )"
   print line[5:-2]
