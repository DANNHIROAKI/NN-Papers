#ifndef __GENERAL_DEFINITION
#define __GENERAL_DEFINITION

#include <stdio.h>
#include <ctype.h>

#define BFHEAD_LENGTH (sizeof(int)*2)    //file header size

typedef char Block[];
//-------------------All-------------------------------------
#define MAXREAL         1e20f
#define FLOATZERO       1e-8f
#define DIMENSION       3


#define min(a, b) (((a) < (b))? (a) : (b)  )
#define max(a, b) (((a) > (b))? (a) : (b)  )


extern int blocksize;

//-------------------Class and other data types--------------
class BlockFile;  //for BlockFile definition
class Cache;
class Cacheable {
	// inherit this class if you wish to use an external memory structure with a cache
public:
	BlockFile *file;
	Cache *cache;
};

enum SECTION {OVERLAP, INSIDE, S_NONE};
enum R_OVERFLOW {SPLIT, REINSERT, NONE};
enum R_DELETE {NOTFOUND,NORMAL,ERASED};

struct SortMbr  {
    int dimension;
    float *mbr;
    float *center;
    int index;
};

//-----Global Functions--------------------------------------
void error(char *_errmsg, bool _terminate);
float area(int dimension, float *mbr);
float margin(int dimension, float *mbr);
float overlap(int dimension, float *r1, float *r2);
float* overlapRect(int dimension, float *r1, float *r2);
bool MBR_section(float *mbr,float* bounces);
float MINDIST_SQR(float *bounces1, float *bounces2);
float MINDIST(float *bounces1, float *bounces2);
float MAXDIST_SQR(float *bounces1, float *bounces2);
float MAXDIST(float *bounces1, float *bounces2);
float MINMAXDIST(float *_p, float *bounces);
void print_mbr(float *mbr,char* msg=NULL);

void enlarge(int dimension, float **mbr, float *r1, float *r2);
int sort_lower_mbr(const void *d1, const void *d2);
int sort_upper_mbr(const void *d1, const void *d2);
int sort_center_mbr(const void *d1, const void *d2);



using namespace std;
#include <vector>
#include <queue>

struct HeapEntry {
	//bool isInside;

	int level,son;
	float key;
	float bounces[2*DIMENSION];
  int end;

	int bk_id;

	// for ECJ only
	bool isPruned;
	float center[2*DIMENSION];
	float radius;
	float tval;	// for holding temporary value
};

struct HeapComp {
	bool operator () (HeapEntry left,HeapEntry right) const
	{ return left.key < right.key; }
};

struct SortComp {
	bool operator () (HeapEntry left,HeapEntry right) const
	{ return left.key < right.key; }
};


template<typename _Tp, typename _Sequence, typename _Compare >
    class FAST_HEAP
    {

    public:
      typedef typename _Sequence::value_type                value_type;
      typedef typename _Sequence::reference                 reference;
      typedef typename _Sequence::const_reference           const_reference;
      typedef typename _Sequence::size_type                 size_type;

    protected:
      _Sequence  c;
      _Compare   comp;

    public:
      explicit
      FAST_HEAP(const _Compare& __x = _Compare(),
		     const _Sequence& __s = _Sequence())
      : c(__s), comp(__x)
      { std::make_heap(c.begin(), c.end(), comp); }

      bool empty() const { return c.empty(); }

      size_type size() const { return c.size(); }

      const_reference top() const {
		return c.front();
      }

      void push(const value_type& __x) {
          c.push_back(__x);
          std::push_heap(c.begin(), c.end(), comp);
      }

      void pop() {
          std::pop_heap(c.begin(), c.end(), comp);
          c.pop_back();
      }
};

//typedef	priority_queue<HeapEntry,vector<HeapEntry>,HeapComp> Heap;
typedef	FAST_HEAP<HeapEntry,vector<HeapEntry>,HeapComp> Heap;

//typedef vector<HeapEntry> EntryList;
typedef vector<HeapEntry> HpeVec;




//-----Polygon Functions--------------------------------------
typedef enum { Pin, Qin, Unknown } tInFlag;
typedef float  tPoint[2];   // type float point
typedef vector<float> FloatVec;
typedef vector<int> IntVec;


// functions for polygon
// P has n vertices, Q has m vertices
bool PolyPolyIntersect(FloatVec& vec, int n, float* P_ptr, int m, float* Q_ptr);
bool PolyMbrIntersect(FloatVec& vec, int n, float* P_ptr, float* bounces);
//void PrintPoly(int n, float* P);
void PrintPoly(FloatVec& vec);
bool InPoly(tPoint q,int n,tPoint* PX);
void AntiClockwiseSort(int n,float* P_ptr);
int	AreaSign( tPoint a, tPoint b, tPoint c );
char SegSegInt( tPoint a, tPoint b, tPoint c, tPoint d, tPoint p, tPoint q);
void getPolyMBR(float* bounces,FloatVec& vec);

bool isPointInsidePolygon(float* pmbr,FloatVec& vec);
bool isBouncesInsidePolygon(float* bounces,FloatVec& vec);

#endif

