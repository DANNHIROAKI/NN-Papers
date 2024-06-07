#			Linux 
#CC       = g++
#CCOPTS   = -c -I/usr/include/X11R5 -I/usr/X11R6/include -I/usr/include/Motif1.2 -g -DLINUX
#LINK     = g++
#LINKOPTS = -static -L/usr/lib/Motif1.2 -L/usr/lib/X11R5 -lm

#			HP-UX
CC       = g++
CCOPTS   = -c -O3
LINK     = g++
LINKOPTS = -lm -g

PACK = functions.o blk_file.o rtree.o hilbert.o

.cc.o:
	$(CC) $(CCOPTS) $<

all: main gendata

# any with gendef.h depends on functions.o

functions.o: functions.cc

blk_file.o: blk_file.cc

rtree.o: rtree.cc

hilbert.o: hilbert.cc

test: test.cc $(PACK) 
	$(LINK) -o test test.cc $(PACK) $(LINKOPTS)

overall: overall.cc $(PACK) 
	$(LINK) -o overall overall.cc $(PACK) $(LINKOPTS)

update: update.cc $(PACK) 
	$(LINK) -o update update.cc $(PACK) $(LINKOPTS)

parameter: parameter.cc $(PACK) 
	$(LINK) -o parameter parameter.cc $(PACK) $(LINKOPTS)


clean:
	rm *.o *stackdump *.exe 
