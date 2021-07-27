"""
Extract small subset of shots from ST10010_1150780_40203.sgy data

Author: M. Ravasi

"""
import segyio

srcfile = '../Data/ST10010_1150780_40203.sgy'

############################################################
# 1. Create small file with traces at start of original file

# Define output file
destfile = 'ST10010_1150780_40203_2dline.sgy'

# Choose subset of interest
nr = 240
nc = 4
ntraces = 20 * nr * nc

# Read input file
src = segyio.open(srcfile, 'r', ignore_geometry=True)
spec = segyio.tools.metadata(src)
spec.tracecount = ntraces
spec.samples = src.samples[:2000]

# Create output file
with segyio.create(destfile, spec) as dst:
	destspec = segyio.tools.metadata(dst)
	print(destspec.tracecount, dst.tracecount)
	print(len(destspec.samples))
	dst.text[0] = src.text[0]
	dst.bin = src.bin
	dst.bin[segyio.BinField.Samples] = 2000
	dst.header = src.header
	dst.trace = src.trace
	print(dst.trace[0])
	print(src.trace[0][:2000]-dst.trace[0])

# Inspect created file 
dst = segyio.open(destfile, 'r', ignore_geometry=True)
print(dst.tracecount, len(dst.samples))

#################################################################
# 2. Create small file with traces in the middle of original file

# Define output file
destfile = 'ST10010_1150780_40203_2dline1.sgy'

# Choose subset of interest
isrcin, isrcend = 4960, 4990
itracein = nr * nc * isrcin
itraceend = nr * nc * isrcend
ntraces = itraceend - itracein

# Read input file
src = segyio.open(srcfile, 'r', ignore_geometry=True)
spec = segyio.tools.metadata(src)
spec.tracecount = ntraces
spec.samples = src.samples[:2000]

# Create output file
with segyio.create(destfile, spec) as dst:
	destspec = segyio.tools.metadata(dst)
	print(destspec.tracecount, dst.tracecount)
	print(len(destspec.samples))
	dst.text[0] = src.text[0]
	dst.bin = src.bin
	dst.bin[segyio.BinField.Samples] = 2000
	for i in range(ntraces):
		dst.header[i] = src.header[i+itracein]
		dst.trace[i] = src.trace[i+itracein]
