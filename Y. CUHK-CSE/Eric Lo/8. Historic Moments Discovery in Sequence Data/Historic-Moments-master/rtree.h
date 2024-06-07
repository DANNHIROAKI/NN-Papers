/* rtree.h
   this file defines the class RTree*/

#ifndef __RTREE
#define __RTREE

#include "gendef.h"
using namespace std;
#include <deque>

class Cache;
class RTNode;
class Entry;
class RTree;


class PolyEntry  {
public:
//--===on disk===--
	int son;
	short vsize;
	float *vertices;	// format: (x0,y0), (x1,y1), (x2,y2), ...   in clockwise format
//--===others===--
	int dimension;
	RTree *my_tree;
	float *bounces;
	double sort_key;	// only used for bulkloading, non-copied

	static int get_min_size();
	int get_size(); 	// dynamic size
	void read_from_buffer(char *buffer);// reads data from buffer
    void write_to_buffer(char *buffer); // writes data to buffer
    void update_bounces();

	void assign_data(FloatVec& dvec);
	void assign_data(PolyEntry* pe);
	void print();

	SECTION section(float *mbr);        // tests, if mbr intersects the box of the object
	bool PolyPolySection(FloatVec& vec,int m, float* Q_ptr);
	bool PolyMbrSection(FloatVec& vec,float* bounces);

	PolyEntry();
	PolyEntry(RTree *rt);
   	 ~PolyEntry();
};

class PolyNode { // a leaf node for storing polygons
public:
//--===on disk===--
	char level;
	int block,num_entries;
	PolyEntry *entries;
//--===others===--
	int dimension;
	bool dirty;
	RTree *my_tree;
//--===functions===--
	PolyNode(RTree *rt);
    PolyNode(RTree *rt,int _block);
    PolyNode(int _cap);	// dummy node
    ~PolyNode();

	float *get_mbr();
	void print();
	bool is_data_node() { return true; };
	void read_from_buffer(char *buffer);
	void write_to_buffer(char *buffer);
};

class Entry  {
public:
//--===on disk===--
	int son,num_data;
	float *bounces;
	////// MODIFICATION BY RAN
	int end;
//--===others===--
	int dimension,level;
    RTree *my_tree;
    RTNode *son_ptr;
    double sort_key;	// only used for bulkloading, non-copied

//--===functions===--
	Entry();
	Entry(int dimension, RTree *rt);
    ~Entry();

	int get_size();
	RTNode *get_son();
	void del_son();
	void init_entry(int _dimension, RTree *_rt);
	void read_from_buffer(char *buffer);// reads data from buffer
    SECTION section(float *mbr);        // tests, if mbr intersects the box
	void write_to_buffer(char *buffer); // writes data to buffer

    virtual Entry & operator = (Entry &_d);
	bool operator == (Entry &_d);
};

class RTNode {
public:
//--===on disk===--
	char level;
	int block,num_entries;
	Entry *entries;
//--===others===--
	bool dirty;
	int capacity,dimension;
	RTree *my_tree;
//--===functions===--
	RTNode(RTree *rt);
    RTNode(RTree *rt,int _block);
    RTNode(int _dim,int _cap);	// dummy node
    ~RTNode();

    int choose_subtree(float *brm);
	R_DELETE delete_entry(Entry *e);
	void enter(Entry *de);
	bool FindLeaf(Entry *e);
	float *get_mbr();
	R_OVERFLOW insert(Entry *d, RTNode **sn);
	bool is_data_node() { return (level==0); };
	void print();
	void read_from_buffer(char *buffer);
	int split(float **mbr, int **distribution);
	void split(RTNode *sn);
    void write_to_buffer(char *buffer);
    void update_count();
};

// an aggregation R-tree
class RTree : public Cacheable {
public:
//--===on disk===--
	int dimension;
	int num_of_data,num_of_dnodes,num_of_inodes;
	int root;
	bool root_is_data;

//--===others===--
	RTNode *root_ptr;
    bool *re_level;
    deque<void*> re_data_cands,deletelist;
	int leaf_acc,non_leaf_acc;

//--===functions===--
	RTree(char *fname,int _b_length,Cache* c,int _dimension);
    RTree(char *fname,float cache_factor);
    ~RTree();
	void del_root();
	bool delete_entry(Entry *d);
	bool FindLeaf(Entry *e);
    int get_num() { return num_of_data; }
	void insert(Entry *d);
	void load_root();
	void read_header(char *buffer);
	void write_header(char *buffer);
};

#endif // __RTREE
