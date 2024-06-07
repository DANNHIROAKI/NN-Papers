#ifndef __BLK_FILE
#define __BLK_FILE


#include <stdio.h>
#include "gendef.h"

class BlockFile {
public:
	FILE* fp;			// os file pointer
	char* filename;		 
	int blocklength;	    // length of a block
	int act_block; 	    // block # of fp's position (fp can stay at block boundaries)
	int number;		    // total # of blocks
	bool new_flag;		// specifies if this is a new file
	
	BlockFile(char* name, int b_length);
	~BlockFile();
	
	void put_bytes(char* bytes,int num) { fwrite(bytes,num,1,fp); }
	void get_bytes(char* bytes,int num) { fread(bytes,num,1,fp); }
	void fwrite_number(int num);	
	int fread_number();
	void seek_block(int bnum) { fseek(fp,(bnum-act_block)*blocklength,SEEK_CUR); }
	
	void read_header(char * header);
	void set_header(char* header);
	bool read_block(Block b,int i);	
	bool write_block(Block b,int i);
	int append_block(Block b);	
	bool delete_last_blocks(int num);
	
	bool file_new()  { return new_flag; }
	int get_blocklength() { return blocklength; }
	int get_num_of_blocks()	{ return number; }
};

class Cacheable;

class Cache {
public:
	enum uses {free,used,fixed};	// for fuf_cont
	int ptr;		        //current position in cache
	int cachesize;		//the number of blocks kept in memory
	int blocklength;
	int page_faults;
	int *cache_cont;	    // array of the indices of blocks that are in cache
	Cacheable **cache_tree;  // array of ptrs to the correspondent Cacheables where the blocks belong to
	uses *fuf_cont; 		//indicator array that shows whether one cache block is free, used or fixed
	int  *LRU_indicator; //indicator that shows how old (unused) is a page in the cache
	bool *dirty_indicator;  //indicator that shows if a cache page has been written
	char **cache;   		// Cache
	
	int next();
	int in_cache(int index, Cacheable *rt);
	Cache(int csize, int blength);
	~Cache();
	bool read_block(Block b, int i, Cacheable *rt);
	bool write_block(Block b, int i, Cacheable *rt);
	bool fix_block(int i, Cacheable *rt);
	bool unfix_block(int i, Cacheable *rt);
	void unfix_all();
	void set_cachesize(int s);
	void flush();			
};

#endif // __BLK_FILE

