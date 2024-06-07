#include "rtree.h"
#include "blk_file.h"
#include "gendef.h"
#include "hilbert.h"
#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <stack>
#include <float.h>
#include <math.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <time.h>
#include <sys/dir.h>
#include <sys/stat.h>

#define DOM_SZ (10000.0)
#define MAXLEVEL (8)
#define RTFILE "Rtree"
#define RTFILE2 "ORtree"
unsigned gBits=12;	// 10
int blk_len=4096;
int LastBlockId=0;
char* gBlock=NULL;
float part, sim_degree, topk_option, delta;
float vmax=-DBL_MAX;
float dmax=-DBL_MAX;
int data_size, data_size2;
int opt; // query option
const float Sentinel=-DBL_MAX;
float topk_low, topk_up;
float sim_low, sim_up;

clock_t t0, t1, t2, t3, t4;

//#define PATH "/Users/lizbai/Documents/Datasets"  //TODS 8 datasets
#define PATH "/Users/lizbai/Documents/data_generator/Datasets"
#define FILENAME "did"


typedef pair<float,float> lvalue; //for candidate streaks ending at position k
struct lrvalue // streak information: (start, end, value) -> (l,r,value)
{
    float l;
    float r;
    float value;
};

struct sjvalue // streak representation for skyline uses : (|s|, j, v) -> (streak length, end, value) -> (s, j, value)
{
	float s;
	float j;
	float value;
};
typedef pair<lrvalue, lrvalue> Oss; // Overpruner and its relevant LPSK pair

void MakeMaximalStreaks(vector<float> data, float part, vector<lrvalue> & MaximalStreaks, vector<lrvalue> & LPSK)
{//for making maximal streaks give dataset
    float k=0;
    stack<lvalue> CandidateStreaks;
    lvalue InitialPair(k,Sentinel);
    CandidateStreaks.push(InitialPair);
    for(int it=0;it<int(data.size()*part);it++)
    {
        k++;
        float s=data[it];
        bool HaveInsert=false;
        while(!CandidateStreaks.empty())
        {
            float v=CandidateStreaks.top().second;
            if(v>s)
            {
                lrvalue TempStreak={CandidateStreaks.top().first,k-1,v};
                CandidateStreaks.pop();
                MaximalStreaks.push_back(TempStreak);
                HaveInsert=true;
            }
            else if(v==s)
                break;
            else
            {
                if(HaveInsert)//some elements were inserted to MaximalStreaks in previous step
                {
                    vector<lrvalue>::iterator it=MaximalStreaks.end();
                    --it;
                    lvalue TempCandidate(it->l,s);
                    CandidateStreaks.push(TempCandidate);
                }
                else//no elements inserted in this iteration
                {
                    lvalue TempCandidate(k,s);
                    CandidateStreaks.push(TempCandidate);
                }
                break;
            }
        }
        if (MaximalStreaks.size() % 100 == 0)
			cout<< " *** here we get --> "<<MaximalStreaks.size()<<endl;
    }
    k+=1;
    while(CandidateStreaks.top().second>Sentinel)
    {
        float v=CandidateStreaks.top().second;
        //lvalue Temp(k,v);
        lrvalue TempStreak={CandidateStreaks.top().first,k-1,v};//Temp);
        CandidateStreaks.pop();
        LPSK.push_back(TempStreak);
    }
}

bool CompareLPSk (lrvalue i, lrvalue j){
    return (i.value>j.value);	
}


void SortLPSk (vector<lrvalue> & _LPSk){
	sort(_LPSk.begin(),_LPSk.end(),CompareLPSk); // sort LPSk in descending value, acsending length
}

vector<Oss> MakeOverPruner(vector<lrvalue> LPSk, vector<lrvalue> &skyline) // Given all LPS and LPSk, find overpruner pairs <overpruner, lpsk>
{
    SortLPSk(LPSk);
    vector<Oss> overprunerPair;
    vector<lrvalue>::iterator ss;    
    vector<lrvalue>::iterator oo;
    vector<int> set;

    for(ss=skyline.begin();ss!=skyline.end();ss++)//traverse
    {
    	float lenth_tp = ss->r - ss->l;
        for(oo=LPSk.begin();oo!=LPSk.end();oo++)
        {
            if((oo->l <= ss->l) && (oo->r >= ss->r) && (lenth_tp >= ((oo->r - oo->l)*sim_degree)))        
        	{
        		//cout<<oo->l<<","<<oo->r<<","<<oo->value<<"******"<<ss->l<<","<<ss->r<<","<<ss->value<<endl;
        		Oss Temp(*ss, *oo);
            	overprunerPair.push_back(Temp);
            	set.push_back(int(ss - skyline.begin()));
            	break;
        	}
    	}
    }
    if(set.size()==0)
        return overprunerPair;
    sort(set.begin(), set.end());
    for(vector<int>::iterator se = set.end()-1;se>=set.begin();se--)
    {
        skyline.erase(skyline.begin()+*se);
    }
    return overprunerPair;
}

void GetDataTable (vector<string> &_Time, vector<float> &_Data, string File)
{//read data and time tables from file
    fstream _file;
    _file.open(File.c_str(),ios::in);
    if(!_file)
    {
        cout<<File<<" does not exist! "<<endl;
        return;
    }
    string s1;
    float s2;
    while(!_file.eof())
    {
        _file>>s1>>s2;
        _Time.push_back(s1);
        _Data.push_back(s2);
    }
    return;
}

void GetDataTable (vector<float> &_Data, string File)
{//read data and time tables from file
    fstream _file;
    _file.open(File.c_str(),ios::in);
    if(!_file)
    {
        cout<<File<<" does not exist! "<<endl;
        return;
    }
    string s1;
    float s2;
    while(!_file.eof())
    {
        _file>>s1>>s2;
        _Data.push_back(s2);
    }
    return;
}

long GetFileSize(char * filename)
{
	FILE *pFile;
	long size;

	pFile = fopen(filename, "rb");
	if(pFile==NULL) perror ("Error opening file");
	else
	{
		fseek(pFile, 0 , SEEK_END);
		size = ftell (pFile);
		fclose(pFile);
	}
	return size;
}

// assume all coordinate values >=0
inline float MIN_FVALUE(float* bounces) {
	float sum=0;
	for (int j=0;j<DIMENSION;j++)
		sum+=bounces[2*j];
	return sum;
}


inline float MAX_FVALUE(float* bounces) {
	float sum=0;
	for (int j=0;j<DIMENSION;j++)
		sum+=bounces[2*j+1];
	return sum;
}

inline bool isDomed(HeapEntry he, Entry* New_Added) 
{// true: he is overlapped/in the domination area of New_Add

		for (int j=0;j<DIMENSION;j++) 
		{
			if (New_Added->bounces[2*j+1] < he.bounces[2*j]) 
			{
				return false;
			}
		}
	return true; 
}


inline bool InSearchRange(HeapEntry he, Entry *Ask)
{//true: he is/contains candidate streaks of Ask
	if ((he.bounces[1] > Ask->bounces[0]*sim_degree) && (he.bounces[2] < Ask->bounces[2]) && (he.bounces[5] > Ask->bounces[4]*sim_degree)){
		return true;}
	else{
		return false;}
}

inline bool IntersectSearchRange(HeapEntry he, Entry *Ask)
{//true: he might contain candidate streaks of Ask
	if ((he.bounces[1] > Ask->bounces[0]*sim_degree) && (he.bounces[2] < Ask->bounces[2]) && (he.bounces[5] > Ask->bounces[4]*sim_degree)){
		return true;}
	else{
		return false;}
}

inline bool JudgePrune(HeapEntry he, vector<sjvalue> skyline)
{
	if (skyline.size() < 1)
		return false;
	for(vector<sjvalue>::iterator it = skyline.begin(); it < skyline.end(); it++ )
	{
		if ((it->s >= he.bounces[1]) && (it->j >= he.bounces[3]) && (it->value >= he.bounces[5]))
			return true; //skip this HeapEntry, it is pruned by skyline points already
	}
	return false;
}

inline vector<Oss> JudgeOverpruner (HeapEntry he, vector<Oss> overprunerPair)// return the sets of overpruners, they dominate HeapEntry he
{
	vector<Oss> Answer;
	if(overprunerPair.size() < 1)
		return Answer;
	for(vector<Oss>::iterator it = overprunerPair.begin(); it < overprunerPair.end(); it++ )
	{
		float length = it->first.r - it->first.l;
		float end = it->first.r;
		float value = it->first.value;
		if ((length >= he.bounces[0]) && (end >= he.bounces[2]) && (value >= he.bounces[4]))
			Answer.push_back(*it);
	}
	return Answer;
}

inline bool JudgeDominate(HeapEntry he, vector<sjvalue> N)
{
	if (N.size() < 1)
		return false;
	for(vector<sjvalue>::iterator it = N.begin(); it < N.end(); it++ )
	{
		if ((it->s >= he.bounces[0]) && (it->j >= he.bounces[2]) && (it->value >= he.bounces[4]))
			return true; //some streaks in N can dominate the HeapEntry he
	}
	return false;
}


int DomRelation(sjvalue a, sjvalue b)
{
	if((a.s >= b.s) && (a.j >= b.j) && (a.value >= b.value))
		return 1;
	else if ((b.s >= a.s) && (b.j >= a.j) && (b.value >= a.value))
		return -1;
	else
		return 0;
}

vector<sjvalue> FindSkyline (vector<sjvalue> Candidate)
{
	vector<sjvalue> Answer;
	if(Candidate.size() < 1)
		return Answer;
	Answer.push_back(*Candidate.begin()); //initialization
	vector<sjvalue>::iterator Candidate_it;
	vector<sjvalue>::iterator Answer_it;

	for(Candidate_it=Candidate.begin()+1; Candidate_it!= Candidate.end(); Candidate_it++)
	{
		bool append = true;
		vector<int> Removal;
		for(Answer_it = Answer.begin(); Answer_it!= Answer.end(); Answer_it++)
		{
			int judge = DomRelation(*Answer_it, *Candidate_it);
			if (judge > 0)
			{
				append = false;
			}
			if (judge < 0)
			{
				Removal.push_back(Answer_it - Answer.begin());
			}
		}
		if (append)
		{
			if(Removal.size()!=0)
			{
				sort(Removal.begin(), Removal.end());
				for(vector<int>::iterator it = Removal.end()-1; it>=Removal.begin(); it--)
					Answer.erase(Answer.begin() + *it);
			}
			Answer.push_back(*Candidate_it);
		}
	}
	return Answer;
}

void PrintResult (vector<sjvalue> Answer, vector<string> Time)
{
	vector<sjvalue>::iterator it;
	if (Answer.size() == 0)
		return;
	cout<<"then number of results is "<<Answer.size()<<endl;
	for (it = Answer.begin(); it!= Answer.end(); it++)
	{
		cout<<"("<<Time[it->j - it->s - 1]<<","<<Time[it->j -1]<<","<<it->value<<"); ";	
	}
	cout<<endl;
	return;
}

vector<sjvalue> QueryBBS(RTree* rt, Entry* Ask) //use BBS to get the HM of ask (in search range of Ask, then skyline)
{
	int count = 0;
	Heap hp;
	while (hp.size()>0) hp.pop();	// clear the heap first
	rt->load_root();
	vector<sjvalue> Answer; 
	sjvalue Temp = {0,0,0};

	{
		RTNode* cur_node=rt->root_ptr;	// enqueue root entries
		for (int i=0;i<cur_node->num_entries;i++)
		{
			HeapEntry he;
			he.key = MAX_FVALUE(cur_node->entries[i].bounces);  // bounces are just MBR or points
			he.level=cur_node->level;
			he.end=cur_node->entries[i].end; // for attribute "end"
			he.son=cur_node->entries[i].son;
			for (int j=0;j<2*DIMENSION;j++)
				he.bounces[j]=cur_node->entries[i].bounces[j];
			if(IntersectSearchRange(he, Ask))
			{
				hp.push(he);
				count++;
			}			
		}
	}

	while (hp.size()>0) {
		HeapEntry he=hp.top();	// dequeue next entry
		//cout<<he.key<<"**";
		hp.pop();
		if(JudgePrune(he, Answer))
			continue;
		if (he.level!= '\0') { //go to its children and insert into the heap
			RTNode *rtn=new RTNode(rt,he.son);
			assert(rtn);
			//printf("%d %d\n",he.level,he.son);
			for (int i=0;i<rtn->num_entries;i++)
			{
				HeapEntry new_he;
				new_he.key = MAX_FVALUE(rtn->entries[i].bounces);
				new_he.level=rtn->level;
				new_he.end=rtn->entries[i].end;//for attribute "end"
				new_he.son=rtn->entries[i].son;
				for (int j=0;j<2*DIMENSION;j++){
					new_he.bounces[j]=rtn->entries[i].bounces[j];}
				if(IntersectSearchRange(new_he, Ask))
				{
					hp.push(new_he);
					count++;
				}
			}
			delete rtn;
			rtn=NULL;
		} 
		else
		{ // Candidate Points
			if(InSearchRange(he, Ask))
			{
				Temp.s = he.bounces[0];
				Temp.j = he.bounces[2];
				Temp.value = he.bounces[4];
				Answer.push_back(Temp);
			//	cout<<"skyline now insert: " <<Temp.j - Temp.s<<" ,"<<Temp.j<<" ,"<<Temp.value<<endl;
			}
		}
	}
	cout<<"Heap Push Times: "<<count<<endl;
	//cout<<"Number of Candidates is "<<Answer.size()<<endl;
	return Answer;
//	printf("\n");
	//printf("rslt: %d,  %d %d\n",results.size(),start_pos,len);
}


vector<sjvalue> GetCandidates(RTree* rt, Entry* Ask) 
{
	Heap hp;
	while (hp.size()>0) hp.pop();	// clear the heap first
	rt->load_root();
	vector<sjvalue> Answer; 
	sjvalue Temp = {0,0,0};

	{
		RTNode* cur_node=rt->root_ptr;	// enqueue root entries
		for (int i=0;i<cur_node->num_entries;i++)
		{
			HeapEntry he;
			he.key=0;  // bounces are just MBR or points
			he.level=cur_node->level;
				he.end=cur_node->entries[i].end; // for attribute "end"
			he.son=cur_node->entries[i].son;
			for (int j=0;j<2*DIMENSION;j++)
				he.bounces[j]=cur_node->entries[i].bounces[j];
			hp.push(he);
		}
	}

	while (hp.size()>0) {
		HeapEntry he=hp.top();	// dequeue next entry
		//cout<<he.key<<"**";
		hp.pop();		
		if (he.level!= '\0') { //go to its children and insert into the heap
			RTNode *rtn=new RTNode(rt,he.son);
			assert(rtn);
			//printf("%d %d\n",he.level,he.son);
			for (int i=0;i<rtn->num_entries;i++)
			{
				HeapEntry new_he;
				new_he.key=0;
				new_he.level=rtn->level;
					new_he.end=rtn->entries[i].end;//for attribute "end"
				new_he.son=rtn->entries[i].son;
				for (int j=0;j<2*DIMENSION;j++){
					new_he.bounces[j]=rtn->entries[i].bounces[j];}
				hp.push(new_he);
			}
			delete rtn;
			rtn=NULL;
		} 
		else
		{ // Candidate Points
			if(InSearchRange(he, Ask))
			{
				Temp.s = he.bounces[0];
				Temp.j = he.bounces[2];
				Temp.value = he.bounces[4];
				Answer.push_back(Temp);
			}
		}
	}
	cout<<"Number of Candidates is "<<Answer.size()<<endl;
	return Answer;
//	printf("\n");
	//printf("rslt: %d,  %d %d\n",results.size(),start_pos,len);
}

void AdvanceUpdateOneData (RTree *rt, float NewData, vector<lrvalue> &LPSK, vector<lrvalue> &BigDelta)
{
	bool judge = true;
	lrvalue MTemp;
	for(vector<lrvalue>::iterator it = LPSK.end()-1; it >= LPSK.begin(); it--)
	{
		if(it->value > NewData)
		{
			if(judge)
			{
				MTemp.l = it->l;
				MTemp.r = it->r + 1;
				MTemp.value = NewData;
			}
			judge = false;
			BigDelta.push_back(*it);
			{// insert into rtree
				Entry * d = new Entry(DIMENSION, NULL);
				d->bounces[0] = d->bounces[1] = it->r - it->l;
				d->bounces[2] = d->bounces[3] = it->r;
				if(DIMENSION == 3)
					d->bounces[4] = d->bounces[5] = it->value;
				d->end=it->r;
				rt->insert(d);
				d=NULL;
			}
			vector<lrvalue>::iterator iit = it;
			LPSK.erase(iit);			
		}
		else
		{
			it->r+=1; //lps_{k+1}
		}
	}
	if(judge)
	{
		lrvalue temp = {LPSK.begin()->r, LPSK.begin()->r, NewData};
		LPSK.insert(LPSK.begin(), temp);
	}
	else
		LPSK.insert(LPSK.begin(), MTemp);
	return;
}

void UpdateOneData (RTree *rt, float NewData, vector<lrvalue> &LPSK)
{
	bool judge = true;
	lrvalue MTemp;
	for(vector<lrvalue>::iterator it = LPSK.end()-1; it >= LPSK.begin(); it--)
	{
		if(it->value > NewData)
		{
			if(judge)
			{
				MTemp.l = it->l;
				MTemp.r = it->r + 1;
				MTemp.value = NewData;
			}	
			{// insert into rtree
				Entry * d = new Entry(DIMENSION, NULL);
				d->bounces[0] = d->bounces[1] = it->r - it->l;
				d->bounces[2] = d->bounces[3] = it->r;
				if(DIMENSION == 3)
					d->bounces[4] = d->bounces[5] = it->value;
				d->end=it->r;
				rt->insert(d);
				d=NULL;
			}
			vector<lrvalue>::iterator iit = it;
			LPSK.erase(iit);			
		}
		else
		{
			it->r+=1; //lps_{k+1}
		}
	}
	if(judge)
	{
		lrvalue temp = {LPSK.begin()->r, LPSK.begin()->r, NewData};
		LPSK.insert(LPSK.begin(), temp);
	}
	else
		LPSK.insert(LPSK.begin(), MTemp);
	return;
}

void Rule(RTree *rt, vector<Oss> overprunerPair) //remove streaks in Rtree if universally dominated by overpruner
{
	if (overprunerPair.size() < 1)
		return;
	Heap hp;
	while (hp.size()>0) hp.pop();	// clear the heap first
	rt->load_root();
	{
		RTNode* cur_node=rt->root_ptr;	// enqueue root entries
		for (int i=0;i<cur_node->num_entries;i++)
		{
			HeapEntry he;
			he.key=MIN_FVALUE(cur_node->entries[i].bounces);  // bounces are just MBR or points
			he.level=cur_node->level;
				he.end=cur_node->entries[i].end; // for attribute "end"
			he.son=cur_node->entries[i].son;
			for (int j=0;j<2*DIMENSION;j++)
				he.bounces[j]=cur_node->entries[i].bounces[j];
			hp.push(he);
		}
	}

	while (hp.size()>0) 
	{
		HeapEntry he=hp.top();	// dequeue next entry
		hp.pop();
		vector<Oss> Answer = JudgeOverpruner (he, overprunerPair);
		if (Answer.size() < 1)
		{
			continue;
		}
		// if not pruned
		if (he.level!= '\0') {
			RTNode *rtn=new RTNode(rt,he.son);
			assert(rtn);
			//printf("%d %d\n",he.level,he.son);
			for (int i=0;i<rtn->num_entries;i++)
				//if (rtn->entries[i].section(qmbr)!=S_NONE)
			{
				HeapEntry new_he;
				new_he.key=MIN_FVALUE(rtn->entries[i].bounces);
				new_he.level=rtn->level;
					new_he.end=rtn->entries[i].end;//for attribute "end"
				new_he.son=rtn->entries[i].son;
				for (int j=0;j<2*DIMENSION;j++)
					new_he.bounces[j]=rtn->entries[i].bounces[j];
				hp.push(new_he);
			}
			delete rtn;
			rtn=NULL;
		} 
		else
		{ // data point in domination 
			for(vector<Oss>::iterator it = Answer.begin(); it < Answer.end(); it++)
			{
				float length = it->second.r - it->second.l;
				float value = it->second.value;
				if((he.bounces[1] < length * sim_degree) || (he.bounces[5] < value *sim_degree))
				{
					Entry* d = new Entry(DIMENSION,NULL);
					assert(d);
					d->son = he.son;
					d->level = he.level;
					d->end= he.end;
					for (int j=0;j<2*DIMENSION;j++)
					{
						d->bounces[j]=he.bounces[j];
					}		
					rt->delete_entry(d);
					//cout<<"delete!***";
					delete d;
					d=NULL;
				}
			}
		}
	}
	return;
}

/*
inline vector<Oss> JudgeOverpruner (HeapEntry he, vector<Oss> overprunerPair)// return the sets of overpruners, they dominate HeapEntry he
{
	vector<Oss> Answer;
	if(overprunerPair.size() < 1)
		return Answer;
	for(vector<Oss>::iterator it = overprunerPair.begin(); it < overprunerPair.end(); it++ )
	{
		float length = it->first.r - it->first.l;
		float end = it->first.r;
		float value = it->first.value;
		if ((length >= he.bounces[0]) && (end >= he.bounces[2]) && (value >= he.bounces[4]))
			Answer.push_back(*it);
	}
	return Answer;
}*/

void OverprunerPruneSkyline(vector<Oss> overprunerPair, vector<sjvalue> &skyline)  // same as function rule, but remove skyline from vectors directly
{
	for (vector<sjvalue>::iterator it = skyline.end()-1; it >= skyline.begin(); it--)
	{
		for (vector<Oss>::iterator ot = overprunerPair.begin(); ot < overprunerPair.end(); ot ++)
		{
			float ot_lenth = ot->first.r - ot->first.l;
			if ((ot_lenth >= it->s) && (ot->first.value >= it->value) && (ot->first.r >= it->j)) // overpruner ot dominate skyline it
			{
				float ot_second_lenth = ot->second.r - ot->second.l;
				float ot_second_value = ot->second.value;
				if ((it->value < ot_second_value * sim_degree) || (it->s < ot_second_lenth * sim_degree)) // it is not HM of ot's dad
				{
					vector<sjvalue>::iterator iit = it;
					skyline.erase(iit);
					//cout<<" >>> Prune !!! ";
					break;					
				}
			}
		}
	}
	return;
}


vector<Oss> RefreshOverpruner (vector<Oss> overprunerPair_, vector<lrvalue> BigDelta, vector<lrvalue> LPSk, vector<sjvalue>& N)
{
	vector<Oss> overprunerPair;
	SortLPSk(LPSk); // sort LPSK in value descending / length ascending
	for (vector<Oss>::iterator it = overprunerPair_.begin(); it != overprunerPair_.end(); it++)
	{
		for (vector<lrvalue>::iterator ot = LPSk.begin(); ot != LPSk.end(); ot ++)
		{
			if (ot->l <= it->first.l)
			{ //find the relevant lpsk of possible overpruner
				float it_lenth = it->first.r - it->first.l; //length of overpruner
				float ot_lenth = ot->r - ot->l; //length of lpsk
				if (it_lenth >= ot_lenth*sim_degree)
				{//yes, it is still overpruner
					Oss Temp(it->first, *ot);
					overprunerPair.push_back(Temp);
				}
				else
				{//no, it is N
					sjvalue Temp = {it->first.r-it->first.l, it->first.r, it->first.value};
					N.push_back(Temp);
				}
				break;
			}
		}
	}
	for (vector<lrvalue>::iterator it = BigDelta.begin(); it != BigDelta.end(); it++)
	{
		for (vector<lrvalue>::iterator ot = LPSk.begin(); ot != LPSk.end(); ot ++)
		{
			if (ot->l <= it->l)
			{ //find the relevant lpsk of possible overpruner
				float it_lenth = it->r - it->l; //length of overpruner
				float ot_lenth = ot->r - ot->l; //length of lpsk
				if (it_lenth >= ot_lenth*sim_degree)
				{//yes, it is still overpruner
					Oss Temp(*it, *ot);
					overprunerPair.push_back(Temp);
				}
				else
				{//no, it is N
					sjvalue Temp = {it->r-it->l, it->r, it->value};
					N.push_back(Temp);
				}
				break;
			}
		}
	}
	return overprunerPair;
}


void Prune(RTree *rt, vector<sjvalue> N) //remove streaks in Rtree if dominated by skyline(Non-overlap)
{
	if (N.size() < 1)
		return;
	Heap hp;
	while (hp.size()>0) hp.pop();	// clear the heap first
	rt->load_root();
	{
		RTNode* cur_node=rt->root_ptr;	// enqueue root entries
		for (int i=0;i<cur_node->num_entries;i++)
		{
			HeapEntry he;
			he.key=MIN_FVALUE(cur_node->entries[i].bounces);  // bounces are just MBR or points
			he.level=cur_node->level;
				he.end=cur_node->entries[i].end; // for attribute "end"
			he.son=cur_node->entries[i].son;
			for (int j=0;j<2*DIMENSION;j++)
				he.bounces[j]=cur_node->entries[i].bounces[j];
			hp.push(he);
		}
	}

	while (hp.size()>0) 
	{
		HeapEntry he=hp.top();	// dequeue next entry
		hp.pop();
		if (!JudgeDominate(he, N))
		{//judgedominate = false => he could not be pruned by N
			continue;
		}
		// if not pruned
		if (he.level!= '\0') {
			RTNode *rtn=new RTNode(rt,he.son);
			assert(rtn);
			//printf("%d %d\n",he.level,he.son);
			for (int i=0;i<rtn->num_entries;i++)
				//if (rtn->entries[i].section(qmbr)!=S_NONE)
			{
				HeapEntry new_he;
				new_he.key=MIN_FVALUE(rtn->entries[i].bounces);
				new_he.level=rtn->level;
					new_he.end=rtn->entries[i].end;//for attribute "end"
				new_he.son=rtn->entries[i].son;
				for (int j=0;j<2*DIMENSION;j++)
					new_he.bounces[j]=rtn->entries[i].bounces[j];
				hp.push(new_he);
			}
			delete rtn;
			rtn=NULL;
		} 
		else
		{ // data point in domination 
					Entry* d = new Entry(DIMENSION,NULL);
					assert(d);
					d->son = he.son;
					d->level = he.level;
					d->end= he.end;
					for (int j=0;j<2*DIMENSION;j++)
					{
						d->bounces[j]=he.bounces[j];
					}		
					rt->delete_entry(d);
					//cout<<"delete!***";
					delete d;
					d=NULL;
		}
	}
	return;
}


/*
vector<int> UpdateTree(RTree *rt, vector<lrvalue> OverPruner, vector<lrvalue> LPSK)
{
	vector<int> Removal;
	vector<lrvalue>::iterator oo, ll;
	for(oo=OverPruner.begin(); oo!=OverPruner.end(); oo++)
	{
		for(ll=LPSK.begin(); ll!=LPSK.end(); ll++)
		{
			if(ll->l <= oo->l)
				break;
		}
		Rule(rt, *oo, *ll);
		if((ll->r - ll->l) * sim_degree > (oo->r - oo->l))// oo is not a overpruner
			Removal.push_back(oo - OverPruner.begin());
	}
	return Removal;
}

void OverPrunerUpdate(vector<lrvalue> &OverPruner, vector<int> Removal)
{
	if(Removal.size()!=0)
	{
		sort(Removal.begin(), Removal.end());
		for(vector<int>::iterator it = Removal.end()-1; it>=Removal.begin(); it--)
			OverPruner.erase(OverPruner.begin() + *it);
	}
}
*/
///////////////////////above are new functions from Ran////////////////////////////


void RT_addnode(RTree *rt,int *top_level,int capacity,int level,Entry *node_cover,RTNode** cur_node) {
	if (cur_node[level]->num_entries==0) { //new node to be created
        cur_node[level]->dirty=false;
        if ((*top_level)<level) //new root
        	*top_level=level;
        cur_node[level]->level=(char)level;	//init. cur_node[level]
    }

    cur_node[level]->entries[cur_node[level]->num_entries]=*node_cover;
	cur_node[level]->num_entries++;

    if (cur_node[level]->num_entries == capacity) { //node is full
        Entry sup_cover(DIMENSION,rt);

        // write the node back to disk
        if (level>0)
        	rt->num_of_inodes++;
        else
        	rt->num_of_dnodes++;

        cur_node[level]->write_to_buffer(gBlock);
        LastBlockId = rt -> file -> append_block(gBlock);
        //printf("write block %d\n",LastBlockId);

        // set MBR and son ptr
        sup_cover.son = LastBlockId;
        sup_cover.num_data=0;
        for (int i=0;i<cur_node[level]->num_entries;i++) {
        	sup_cover.num_data+=cur_node[level]->entries[i].num_data;
        	float* ref_bounces=cur_node[level]->entries[i].bounces;
        	for (int j=0;j<DIMENSION;j++) {
        		if (i==0) {
        			sup_cover.bounces[2*j]=ref_bounces[2*j];
        			sup_cover.bounces[2*j+1]=ref_bounces[2*j+1];
        		} else {
        			sup_cover.bounces[2*j]=min(sup_cover.bounces[2*j],ref_bounces[2*j]);
	    			sup_cover.bounces[2*j+1]=max(sup_cover.bounces[2*j+1],ref_bounces[2*j+1]);
        		}
        	}
        }

        cur_node[level]->num_entries=0;	// empty cur_node[level] after all updates!
        RT_addnode(rt, top_level, capacity, level+1, &sup_cover, cur_node);
    }
}

RTree *RT_bulkload(RTree *rt, Entry **objs, int count) {
    int top_level=0;
    const float util_rate=0.7;	// typical util. rate=0.7

	int capacity = rt->root_ptr->capacity;	// all nodes have the same capacity
    printf("capacity=%d\n",capacity);

    RTNode** cur_node=new RTNode*[MAXLEVEL];
    for (int i=0; i<MAXLEVEL; i++)
        cur_node[i]=new RTNode(DIMENSION,capacity);

	capacity=(int)(util_rate*capacity);
    printf("util_cap=%d\n", capacity);

	// assume objs sorted based on key
    rt->num_of_data = count;
    rt->num_of_dnodes = 0;

	printf("start\n");

    for (int i=0;i<count;i++) {
        RT_addnode(rt,&top_level,capacity,0,objs[i],cur_node);
        objs[i]=NULL;
        if (i % 10 == 0)
			printf("\rinserting record %d",i);
	}
	printf("\n");

    //flush non-empty blocks
    int level=0;
    while (level<=top_level) {
    	printf("level: %d %d\n",level,top_level);
    	if (cur_node[level]!=NULL) {
	        if (level>0)
	        	rt->num_of_inodes++;
	        else
	        	rt->num_of_dnodes++;

		    if (level<top_level) {
	        	Entry sup_cover(DIMENSION,rt);

		        // write the node back to disk
		        cur_node[level]->write_to_buffer(gBlock);
		        LastBlockId = rt -> file -> append_block(gBlock);

		        // set MBR and son ptr
		        sup_cover.son = LastBlockId;
		        sup_cover.num_data=0;
		        for (int i=0;i<cur_node[level]->num_entries;i++) {
		        	sup_cover.num_data+=cur_node[level]->entries[i].num_data;
		        	float* ref_bounces=cur_node[level]->entries[i].bounces;
		        	for (int j=0;j<DIMENSION;j++) {
		        		if (i==0) {
		        			sup_cover.bounces[2*j]=ref_bounces[2*j];
		        			sup_cover.bounces[2*j+1]=ref_bounces[2*j+1];
		        		} else {
		        			sup_cover.bounces[2*j]=min(sup_cover.bounces[2*j],ref_bounces[2*j]);
			    			sup_cover.bounces[2*j+1]=max(sup_cover.bounces[2*j+1],ref_bounces[2*j+1]);
		        		}
		        	}
		        }

		        cur_node[level]->num_entries=0;	// empty cur_node[level] after all updates!
	        	RT_addnode(rt,&top_level,capacity,level+1,&sup_cover,cur_node);
	        } else {	// root
	            rt->root_ptr->dirty=false;
	            rt->del_root();	// clear old root

	            // write new root
	            cur_node[level]->write_to_buffer(gBlock);
		        rt->file->write_block(gBlock, rt->root);
	            if (level>0)
	            	rt->root_is_data=false;
	        }
	    }
	    level++;
    }
    return rt;
}

int sort_tmpvalue(const void *d1, const void *d2) {
    Entry *s1=*((Entry **) d1), *s2=*((Entry **) d2);
    float diff=s1->sort_key-s2->sort_key;
    //printf("%f\n",diff);
    if (diff<0)
        return -1;
    else if (diff>0)
        return 1;
    else
    	return 0;
}

void BulkLoadData(Entry** dataAry,int data_size,RTree* rt) {
	printf("Create the tree by bulk-loading\n");
	unsigned cDOM=1<<gBits;
	bitmask_t coord[DIMENSION];

	for (int i=0;i<data_size;i++) {
		for (int j=0;j<DIMENSION;j++) {
			coord[j]=(bitmask_t)( ((float) cDOM)*(dataAry[i]->bounces[2*j]/(DOM_SZ+1.0)) );
			//printf("%f\n",(float)(coord[j]));	printf("%f\n",(float)(cDOM));
			assert(0<=coord[j]&&coord[j]<cDOM);
		}

		bitmask_t hrt_value=hilbert_c2i(DIMENSION,gBits,coord);
		dataAry[i]->sort_key=(float)hrt_value;
		assert(dataAry[i]->sort_key>=0);
		if (i % 100 == 0)
			printf("\rcomputing record %d",i);
	}
	printf("\nbegin sorting\n");


	qsort(dataAry,data_size,sizeof(Entry*),sort_tmpvalue);	// for testing
	printf("sorted %d rect.\n", data_size);

	gBlock = new char[blk_len];
	RT_bulkload(rt,dataAry,data_size);

	rt->write_header(gBlock);
    rt->file->set_header(gBlock);
	printf("This R-Tree contains %d internal, %d data nodes and %d data\n",
		   	rt->num_of_inodes, rt->num_of_dnodes, rt->num_of_data);

	delete[] gBlock;
}

void RepeatInsertion(Entry** dataAry,int data_size,RTree* rt) {
	//printf("Create the tree by repeated insertions\n");
	for (int i=0;i<data_size;i++) {
		rt->insert(dataAry[i]); // entry deleted inside insertion function
		dataAry[i]=NULL;

		//if (i % 100 == 0)
			//printf("\rinserting record %d",i);
	}
	//printf("\n");
}

void gen_syn_data(Entry** dataAry,int data_size) {
	float pt[DIMENSION];

	for (int i=0;i<data_size;i++) {
		for (int j=0;j<DIMENSION;j++)	// UI distribution
			pt[j]=drand48();

		for (int j=0;j<DIMENSION;j++) {
			assert(pt[j]>=0&&pt[j]<=1.0);

			pt[j]=pt[j]*DOM_SZ;
			dataAry[i]->bounces[2*j]=dataAry[i]->bounces[2*j+1]=pt[j];
			//printf("%f ",pt[j]);
		}
		//printf("\n");
	}
}

void write_dtfile(char* dtfile,Entry** dataAry,int data_size) {
	FILE* fout=fopen(dtfile,"wb");

	int dim=DIMENSION;
	fwrite(&(dim),1,sizeof(int),fout);
	fwrite(&(data_size),1,sizeof(int),fout);
	for (int i=0;i<data_size;i++)
		fwrite(dataAry[i]->bounces,2*DIMENSION,sizeof(float),fout);
	fclose(fout);
}


void read_real_data(char* rawfn,Entry** dataAry,int& data_size) {
	FILE* fin=fopen(rawfn,"r");

	assert(DIMENSION==2);

	int rec_id=0;
	float xmin,ymin,xmax,ymax;
	float xval,yval;
	float lastx=-1.0,lasty=-1.0;

	xmin=ymin=MAXREAL;
	xmax=ymax=-MAXREAL;

	while (!feof(fin)) {
		fscanf(fin,"%f %f\n",&xval,&yval);

		xmin=min(xval,xmin);
		xmax=max(xval,xmax);

		ymin=min(yval,ymin);
		ymax=max(yval,ymax);


		// remove adjacent duplicaates
		if (xval==lastx&&yval==lasty)
			continue;
		lastx=xval;
		lasty=yval;


		if (rec_id>=data_size) {
			printf("error data size\n");
			exit(0);
		}

		Entry* cur_data=dataAry[rec_id];
		cur_data->bounces[0]=cur_data->bounces[1]=xval;
		cur_data->bounces[2]=cur_data->bounces[3]=yval;

		if (rec_id%10000==0)
			printf("rec %d ok (%f %f)\n",rec_id,xval,yval);

		rec_id++;
		if (rec_id>data_size) {
			printf("input size invalid\n");
			exit(0);
		}
	}
	fclose(fin);

	printf("orig. map: [%f %f] [%f %f]\n",xmin,xmax,ymin,ymax);
	printf("|rec|: %d\n",rec_id);
	data_size=rec_id;

	for (int i=0;i<data_size;i++) {
		Entry* cur_data=dataAry[i];
		float x=cur_data->bounces[0];
		float y=cur_data->bounces[2];

		x=(x-xmin)/(xmax-xmin)*DOM_SZ;
		y=(y-ymin)/(ymax-ymin)*DOM_SZ;

		cur_data->bounces[0]=cur_data->bounces[1]=x;
		cur_data->bounces[2]=cur_data->bounces[3]=y;
	}
}

void getRandomParameter(float &sim_degree, float &topk_option)
{
	sim_degree = ((double)(rand())/((RAND_MAX)))*(sim_up - sim_low) + sim_low;
	topk_option = (int)rand()%int((topk_up - topk_low + 1)) + topk_low;
	return;
}


int main(int argc, char* argv[]) {
	if(argc!=2){
		cout<<"error"<<endl;
		return 0;
	}

	const int total_run_times = 5;
	int  runtime[total_run_times] = {10, 20, 30, 40, 50};

	//opt = atoi(argv[1]); // opt:: algorithm type  1:offline ; 2 : base increment ; 3: minimal increment
	//topk_option = 3;//baseline or optimized 

	topk_low = 1;
	topk_up = 10;
	sim_low = 0.5;
	sim_up = 1.0;

	part = 1;
	blk_len=1024;	// default: 1 K page size (better for join)
	//sim_degree = 0.8; //minimum sim_degree
	

	string path = PATH;
	DIR *pdir;  
    struct dirent *pdirent;  
    char temp[256];
    char tp[256];
    try {  
        pdir = opendir(path.c_str());  
    }  
    catch(const char *str)  
    {printf("failed open dir");} 

    if(!pdir)
    	return -1;

    {
    	int column = 1; 
    	ofstream fout(argv[1]);
    	fout<<"SOIA\tInitialization\tLook-up\tTime\tSpace"<<endl;

    	while((pdirent = readdir(pdir)))
        {    
            if(pdirent->d_name[0]=='.')
            	continue;
            {
				vector<string> Ttable;
				vector<float> Data;
				sprintf(temp, "%s/%s", path.c_str(), pdirent->d_name);
				GetDataTable( Data, temp);
				vector<lrvalue> MaxStreaks, LPSK;
				srand (unsigned(time(NULL)));  //rand-seed

				{
					sim_degree = sim_low; // for initialization
					t0 = clock();
					MakeMaximalStreaks(Data, part, MaxStreaks, LPSK);
					vector<Oss> overprunerPair = MakeOverPruner (LPSK, MaxStreaks);
            		data_size = MaxStreaks.size();

            		remove(RTFILE);// Rtree for non-overlap
					Cache *c=new Cache(10000,blk_len);
					RTree* rt=new RTree(RTFILE, blk_len,c,DIMENSION);
					Entry** dataAry=new Entry*[data_size];

            		for (int i=0;i<data_size;i++) 
            		{
            			Entry* d = new Entry(DIMENSION,NULL);
            			assert(d);
            			d->num_data=1;
            			d->son=i;
            			d->bounces[0] = d->bounces[1] = MaxStreaks[i].r - MaxStreaks[i].l;
            			d->bounces[2] = d->bounces[3] = MaxStreaks[i].r;
            			if(DIMENSION == 3)
            				d->bounces[4] = d->bounces[5] = MaxStreaks[i].value;
            			d->end=MaxStreaks[i].r;
            			dataAry[i]=d;            
            		}
            		RepeatInsertion(dataAry,data_size,rt);

            		Entry *origin = new Entry(DIMENSION, NULL);
					origin->bounces[0] = origin->bounces[1] = 0;
					origin->bounces[2] = origin->bounces[3] = DBL_MAX;
					origin->end = DBL_MAX;
					if(DIMENSION == 3)
						origin->bounces[4] = origin->bounces[5] = 0;
					vector<sjvalue> skyline = QueryBBS(rt, origin);
					Prune(rt, skyline);
            		Rule(rt, overprunerPair);
            		t1 = clock();

            		SortLPSk(LPSK);
            		t2 = clock();
            		for(int k = 0; k < total_run_times; k++) // five cases
					{
						int many = runtime[k];
						for (int j = 0; j < many; j++)
						{
							getRandomParameter(sim_degree, topk_option);
							if(LPSK.size() < topk_option)
									topk_option = LPSK.size();
							for( int i=0; i<topk_option; i++)
							{
								Entry *ask = new Entry(DIMENSION, NULL);
								ask->bounces[0] = ask->bounces[1] = LPSK[i].r - LPSK[i].l;
								ask->bounces[2] = ask->bounces[3] = LPSK[i].r;
								ask->end = LPSK[i].r;
								if(DIMENSION == 3)
									ask->bounces[4] = ask->bounces[5] = LPSK[i].value;
								vector<sjvalue> Answer = QueryBBS(rt, ask);
								delete ask;
							}
						}
					}
					t3=clock(); // remember to divide total_run_times when output
					long tree_size = GetFileSize(RTFILE);
					fout<<column++<<"\t"<<1000*double(t1-t0)/CLOCKS_PER_SEC<<"\t"<<1000*double(t3-t2)/(total_run_times * CLOCKS_PER_SEC)<<"\t"<<1000*double(t1-t0)/CLOCKS_PER_SEC + 1000*double(t3-t2)/(total_run_times * CLOCKS_PER_SEC)<<"\t"<<tree_size<<endl;
				}
			}
		}
		fout.close();
	}
	return 0;
}
