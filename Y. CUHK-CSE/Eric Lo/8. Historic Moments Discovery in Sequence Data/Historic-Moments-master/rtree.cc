#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "rtree.h"
#include "blk_file.h"


PolyEntry::PolyEntry() {
	my_tree=NULL;
	dimension=2;
	bounces=new float[4];

	son=-1;
	vertices=NULL;
	vsize=0;
}

PolyEntry::PolyEntry(RTree *rt) {
	my_tree=rt;
	dimension=2;
	bounces=new float[4];

	son=-1;
	vertices=NULL;
	vsize=0;
}

PolyEntry::~PolyEntry() {
	if (bounces!=NULL) {
		delete[] bounces;
		bounces=NULL;
	}
	if (vertices!=NULL) {
		delete[] vertices;
		vertices=NULL;
	}
}

int PolyEntry::get_size() {
	return 2*vsize*sizeof(float) +  sizeof(int) + sizeof(short);
}

int PolyEntry::get_min_size() {
	return 2*3*sizeof(float) +  sizeof(int) + sizeof(short);
}


void PolyEntry::update_bounces() {
	bounces[0]=bounces[1]=vertices[0];
	bounces[2]=bounces[3]=vertices[1];
	for (int j=0;j<vsize;j++) {
		bounces[0] = min(bounces[0], vertices[2*j]);
		bounces[1] = max(bounces[1], vertices[2*j]);
		bounces[2] = min(bounces[2], vertices[2*j+1]);
		bounces[3] = max(bounces[3], vertices[2*j+1]);
    }
}


void PolyEntry::assign_data(FloatVec& dvec) { // format: sequence of (x,y) coordinates
	if (vertices!=NULL) {	// prevent memory leak
		delete[] vertices;
		vertices=NULL;
	}

	vsize=dvec.size()/2;
	vertices=new float[2*vsize];	// depending on the value of vsize
    for (int j=0;j<dvec.size();j++) // j \in [0,2*vsize)
    	vertices[j]=dvec[j];

  	AntiClockwiseSort(vsize,vertices);
    update_bounces();	// compute mbr of current polygon
}


void PolyEntry::assign_data(PolyEntry* pe) {
	if (vertices!=NULL) {	// prevent memory leak
		delete[] vertices;
		vertices=NULL;
	}

	vsize=pe->vsize;
	vertices=new float[2*vsize];	// depending on the value of vsize
    for (int j=0;j<2*vsize;j++) // j \in [0,2*vsize)
    	vertices[j]=pe->vertices[j];

  	AntiClockwiseSort(vsize,vertices);
    update_bounces();	// compute mbr of current polygon

    son=pe->son;
}

void PolyEntry::print() {
	printf("Polygon %d: [%f %f] [%f %f]\n",son,bounces[0],bounces[1],bounces[2],bounces[3]);
	printf("i\tx\ty\n");
	for (int i = 0; i < vsize; i++ )
		printf("%d\t%f\t%f\n", i, vertices[2*i], vertices[2*i+1]);
}



bool PolyEntry::PolyPolySection(FloatVec& vec,int m, float* Q_ptr) {
	return PolyPolyIntersect(vec,vsize,vertices,m,Q_ptr);
}

bool PolyEntry::PolyMbrSection(FloatVec& vec,float* bounces) {
	return PolyMbrIntersect(vec,vsize,vertices,bounces);
}


SECTION PolyEntry::section(float *mbr) {
    bool inside=true,overlap=true;
    for (int i = 0; i < dimension; i++) {
		if (mbr[2 * i] > bounces[2 * i + 1] ||  mbr[2 * i + 1] < bounces[2 * i])
			overlap = false;
		if (mbr[2 * i] < bounces[2 * i] ||
			mbr[2 * i + 1] > bounces[2 * i + 1])
			inside = false;
    }
    if (inside)
		return INSIDE;
    else if (overlap)
		return OVERLAP;
    else
		return S_NONE;
}


void PolyEntry::read_from_buffer(char *buffer) {
	if (vertices!=NULL) {	// prevent memory leak
		delete[] vertices;
		vertices=NULL;
	}

	int i=0;
    memcpy(&son, &buffer[i], sizeof(int));
    i += sizeof(int);
    memcpy(&vsize, &buffer[i], sizeof(short));
    i += sizeof(short);

    vertices=new float[2*vsize];	// depending on the value of vsize
    memcpy(vertices, &buffer[i], 2*vsize*sizeof(float));
    i += 2*vsize*sizeof(float);

    update_bounces();	// compute mbr of current polygon
}

void PolyEntry::write_to_buffer(char *buffer) {
	int i=0;
    memcpy(&buffer[i], &son, sizeof(int));
    i += sizeof(int);
    memcpy(&buffer[i], &vsize, sizeof(short));
    i += sizeof(short);
    memcpy(&buffer[i], vertices, 2*vsize*sizeof(float));
    i += 2*vsize*sizeof(float);
}


PolyNode::PolyNode(RTree *rt) {	// create a new node on disk
    dimension=2;
    my_tree = rt;
    num_entries = 0;
	dirty = true;

	int header_size = sizeof(char) + sizeof(int);  // level + num_entries
	int temp_cap=(rt->file->get_blocklength() - header_size) / PolyEntry::get_min_size();
    entries = new PolyEntry[temp_cap];

	//assign a new block on the disk
    char *b = new char[rt -> file -> get_blocklength()];
    block = rt -> file -> append_block(b);
    delete [] b;
}

PolyNode::PolyNode(RTree *rt, int _block) {	// restore a node from the disk.
	dimension=2;
	my_tree = rt;
    num_entries = 0;
	dirty = false;

	int header_size = sizeof(char) + sizeof(int);  // level + num_entries
	int temp_cap=(rt->file->get_blocklength() - header_size) / PolyEntry::get_min_size();
    entries = new PolyEntry[temp_cap];

    block = _block;
    char *b = new char[rt -> file -> get_blocklength()];
    if (rt -> cache == NULL) // no cache
        rt -> file -> read_block(b, block);
    else
        rt -> cache -> read_block(b, block, rt);

    read_from_buffer(b);

    delete [] b;

    rt->leaf_acc++;	// must be data node
}

PolyNode::PolyNode(int _cap) {	// dummy node
	dimension=2;
	my_tree = NULL;
    num_entries = 0;
	dirty = false;

    entries = new PolyEntry[_cap];
    block = 0;
}

PolyNode::~PolyNode() {
    if (dirty) {
    	char *b = new char[my_tree->file->get_blocklength()];
        write_to_buffer(b);
        if (my_tree->cache == NULL) // no cache
            my_tree->file->write_block(b, block);
        else
            my_tree->cache->write_block(b, block, my_tree);
        delete [] b;
    }
    delete [] entries;
}

float* PolyNode::get_mbr() {
	float *mbr= new float[2*dimension];
    for (int i = 0; i < 2*dimension; i ++ )
        mbr[i] = entries[0].bounces[i];

    for (int j = 1; j < num_entries; j++) {
    	for (int i = 0; i < 2*dimension; i += 2) {
    	    mbr[i]   = min(mbr[i],   entries[j].bounces[i]);
    	    mbr[i+1] = max(mbr[i+1], entries[j].bounces[i+1]);
        }
    }
    return mbr;
}

void PolyNode::print() {
	printf("level %d  Block: %d\n", level, block);
    for (int i = 0; i < num_entries ; i++) {
        printf("(%4.1lf, %4.1lf, %4.1lf, %4.1lf)\n",
	       entries[i].bounces[0],entries[i].bounces[1],
	       entries[i].bounces[2],entries[i].bounces[3]);
    }
}

void PolyNode::read_from_buffer(char *buffer) {
	int i, j;
    // Level
    memcpy(&level, buffer, sizeof(char));
    j = sizeof(char);

    // num_entries
    memcpy(&num_entries, &(buffer[j]), sizeof(int));
    j += sizeof(int);

    for (i = 0; i < num_entries; i++) {
    	entries[i].read_from_buffer(&buffer[j]);
    	j += entries[i].get_size();
    }
}

void PolyNode::write_to_buffer(char *buffer) {
	int i, j, s;

    // Level
    memcpy(buffer, &level, sizeof(char));
    j = sizeof(char);

    // num_entries
    memcpy(&buffer[j], &num_entries, sizeof(int));
    j += sizeof(int);

    for (i = 0; i < num_entries; i++) {
    	entries[i].write_to_buffer(&buffer[j]);
    	j += entries[i].get_size();
    }
}

Entry::Entry() {
  	// does nothing.  remember you should call init_entry to initialize afterwards
	son_ptr = NULL;
	bounces = NULL;
	num_data=0;
}

Entry::Entry(int _dimension, RTree *rt) {
    dimension = _dimension;
    my_tree = rt;
    bounces = new float[2*dimension];
    son_ptr = NULL;
    son = 0;
	level = 0;
	num_data=0;
    end=0;
}

Entry::~Entry() {
    if (bounces) delete [] bounces;
    if (son_ptr != NULL) delete son_ptr;
}

void Entry::del_son() {
	if (son_ptr != NULL) {
		delete son_ptr;
		son_ptr = NULL;
	}
}

int Entry::get_size() {	//for bounces and son
    return 2 * dimension * sizeof(float) + 3 * sizeof(int);
}

RTNode* Entry::get_son() {
	if (son_ptr == NULL)
	    son_ptr = new RTNode(my_tree, son);
    return son_ptr;
}

void Entry::init_entry(int _dimension, RTree *_rt) {
	dimension = _dimension;
    my_tree = _rt;
    bounces = new float[2*dimension];
    son_ptr = NULL;
    son = 0;
	level = 0;
	num_data = 0;
    end=0;

}

void Entry::read_from_buffer(char *buffer) {
    int i= 2 * dimension * sizeof(float);
    memcpy(bounces, buffer, i);
    memcpy(&son, &buffer[i], sizeof(int));
    i += sizeof(int);
    memcpy(&num_data, &buffer[i], sizeof(int));
    i += sizeof(int);
    ////MODIFIED BY RAN
    memcpy(&end, &buffer[i], sizeof(int));
    i += sizeof(int);

}

SECTION Entry::section(float *mbr) {
    bool inside=true,overlap=true;
    for (int i = 0; i < dimension; i++) {
		if (mbr[2 * i] > bounces[2 * i + 1] ||  mbr[2 * i + 1] < bounces[2 * i])
			overlap = false;
		if (mbr[2 * i] < bounces[2 * i] ||
			mbr[2 * i + 1] > bounces[2 * i + 1])
			inside = false;
    }
    if (inside)
		return INSIDE;
    else if (overlap)
		return OVERLAP;
    else
		return S_NONE;
}

void Entry::write_to_buffer(char *buffer) {
    int i= 2 * dimension * sizeof(float);
    memcpy(buffer, bounces, i);
    memcpy(&buffer[i], &son, sizeof(int));
    i += sizeof(int);
    memcpy(&buffer[i], &num_data, sizeof(int));
    i += sizeof(int);
    ////MODIFIED BY RAN
    memcpy(&buffer[i], &end,sizeof(int));
    i += sizeof(int);

}

bool Entry::operator == (Entry &_d) {
  	//this function compares two entries based on (1)son (2)dimension (3)extents
	if (son != _d.son) return false;
	if (dimension != _d.dimension) return false;
	for (int i = 0; i < 2 * dimension; i++)
		if (fabs(bounces[i] - _d.bounces[i]) > FLOATZERO) return false;
	return true;
}

Entry& Entry::operator = (Entry &_d) {
  	//this function assigns all fieds of _d with the same values of this entry
    dimension = _d.dimension;
    son = _d.son;
    son_ptr = _d.son_ptr;
    memcpy(bounces, _d.bounces, sizeof(float) * 2 * dimension);
    my_tree = _d.my_tree;
	level = _d.level;
	num_data = _d.num_data;
    //// MODIFIED BY RAN

    end=_d.end;

    return *this;
}

RTNode::RTNode(RTree *rt) {	// create a new node on disk
    my_tree = rt;
    dimension = rt->dimension;
    num_entries = 0;
	dirty = true;

    Entry * d = new Entry();
	d -> init_entry(dimension, NULL);
    int header_size = sizeof(char) + sizeof(int);  // level + num_entries
    capacity = (rt -> file -> get_blocklength() - header_size) / d -> get_size();
    delete d;

    entries = new Entry[capacity];
    for (int i = 0; i < capacity; i++)
		entries[i].init_entry(dimension, rt);

	//assign a new block on the disk
    char *b = new char[rt -> file -> get_blocklength()];
    block = rt -> file -> append_block(b);
    delete [] b;
}

RTNode::RTNode(RTree *rt, int _block) {	// restore a node from the disk.
	my_tree = rt;
    dimension = rt->dimension;
    num_entries = 0;
	dirty = false;

    Entry * d = new Entry();
	d -> init_entry(dimension, NULL);
    int header_size = sizeof(char) + sizeof(int);
    capacity = (rt -> file -> get_blocklength() - header_size) / d -> get_size();
    delete d;

    entries = new Entry[capacity];
    for (int i = 0; i < capacity; i++)
		entries[i].init_entry(dimension, rt);

    block = _block;
    char *b = new char[rt -> file -> get_blocklength()];
    if (rt -> cache == NULL) // no cache
        rt -> file -> read_block(b, block);
    else
        rt -> cache -> read_block(b, block, rt);

    read_from_buffer(b);
    delete [] b;

    // recently added 2/8/2005 16:11
    if (is_data_node())
    	rt->leaf_acc++;
    else
    	rt->non_leaf_acc++;
}

RTNode::RTNode(int _dim,int _cap) {	// dummy node
	my_tree = NULL;
    dimension = _dim;
    num_entries = 0;
	dirty = false;

    capacity = _cap;;
    entries = new Entry[capacity];
    for (int i = 0; i < capacity; i++)
		entries[i].init_entry(dimension, NULL);

    block = 0;
}

RTNode::~RTNode() {
    if (dirty) {
    	char *b = new char[my_tree->file->get_blocklength()];
        write_to_buffer(b);
        if (my_tree->cache == NULL) // no cache
            my_tree->file->write_block(b, block);
        else
            my_tree->cache->write_block(b, block, my_tree);
        delete [] b;
    }
    delete [] entries;
}

int RTNode::choose_subtree(float *mbr) {
    int i, j, follow, minindex, *inside, inside_count, *over;
    float *bmbr, old_o, o, omin, a, amin, f, fmin;

    inside_count = 0;
    inside = new int[num_entries];
    over = new int[num_entries];
    for (i = 0; i < num_entries; i++) {
    	switch (entries[i].section(mbr))
    	{
        	case INSIDE:
        	    inside[inside_count++] = i;
        	    break;
        }
    }

    if (inside_count == 1)
        // Case 1: There is exactly one dir_mbr that contains mbr
    	follow = inside[0];
    else if (inside_count > 1)
    // Case 2: There are many dir_mbrs that contain mbr
    // choose the one with the minimum area
    {
    	fmin = MAXREAL;
    	for (i = 0; i < inside_count; i++) {
    	    f = area(dimension, entries[inside[i]].bounces);
    	    if (f < fmin) {
    	    	minindex = i;
          		fmin = f;
       	    }
       	}
    	follow = inside[minindex];
    }
    else
    // Case 3: There are no dir_mbrs that contain mbr
    // choose the one for which insertion causes the minimun overlap if son_is_data
    // else choose the one for which insertion causes the minimun area enlargement
    {
       	if (level == 1) { // son_is_data
            omin = MAXREAL;
    	    fmin = MAXREAL;
    	    amin = MAXREAL;
    	    for (i = 0; i < num_entries; i++) {
        		enlarge(dimension, &bmbr, mbr, entries[i].bounces);

        		// calculate area and area enlargement
        		a = area(dimension, entries[i].bounces);
        		f = area(dimension, bmbr) - a;

        		// calculate overlap before enlarging entry_i
        		old_o = o = 0.0;
        		for (j = 0; j < num_entries; j++) {
        		    if (j != i) {
    			        old_o += overlap(dimension,
    					 entries[i].bounces,
    					 entries[j].bounces);
    			        o += overlap(dimension,
    				     bmbr,
    				     entries[j].bounces);
    		        }
    	        }
    	        o -= old_o;

    	        // is this entry better than the former optimum ?
    	        if ((o < omin) ||
    		    	(o == omin && f < fmin) ||
    		    	(o == omin && f == fmin && a < amin))
    	        {
    	       	    minindex = i;
        		    omin = o;
        		    fmin = f;
        		    amin = a;
        	    }
    	        delete [] bmbr;
    	    }
        } else { // son is not a data node
    	    fmin = MAXREAL;
    	    amin = MAXREAL;
    	    for (i = 0; i < num_entries; i++) {
    	        enlarge(dimension, &bmbr, mbr, entries[i].bounces);

    	        // calculate area and area enlargement
    	        a = area(dimension, entries[i].bounces);
    	        f = area(dimension, bmbr) - a;

    	        // is this entry better than the former optimum ?
    	        if ((f < fmin) || (f == fmin && a < amin)) {
    	       	    minindex = i;
    		        fmin = f;
    	            amin = a;
    	        }
	            delete [] bmbr;
	        }
        }
    	follow = minindex;
    	dirty = true;
    }
    delete [] inside;
    delete [] over;
    return follow;
}

R_DELETE RTNode::delete_entry(Entry *e) {
	RTNode *succ;
	float *tmp;
	if (level > 0) {
		if (this == my_tree->root_ptr) {	//i.e. this is the root
			for (int i = 0; i < num_entries; i++) {
				tmp = overlapRect(dimension, entries[i].bounces, e -> bounces);
				if (tmp != NULL) {
					delete [] tmp;
					succ = entries[i].get_son();
					R_DELETE del_ret;
					del_ret = succ -> delete_entry(e);
					if (del_ret != NOTFOUND) {
						switch (del_ret)
						{
							case NORMAL:
							{	float *mbr= succ -> get_mbr();
								memcpy(entries[i].bounces, mbr, sizeof(float) * 2 * dimension);
								dirty = true;
								delete [] mbr;

								delete entries[i].son_ptr;
								entries[i].son_ptr = NULL;

								return NORMAL;
								break;
							}
							case ERASED:
							{	delete entries[i].son_ptr;
								entries[i].son_ptr = NULL;
								int j;
								for (j = i; j < num_entries - 1; j++)
									entries[j] = entries[j+1];
								for (j = num_entries - 1; j < capacity; j++)
									entries[j].son_ptr = NULL;

								num_entries--;
								dirty = true;
								return NORMAL;
								break;
							}
						}
					}
				}
			}
			return NOTFOUND;
		} else { //is not root and not leaf
			for (int i = 0; i < num_entries; i++) {
				tmp = overlapRect(dimension, entries[i].bounces, e -> bounces);
				if (tmp != NULL) {
					delete [] tmp;
					succ = entries[i].get_son();
					R_DELETE del_ret;
					del_ret = succ->delete_entry(e);
					if (del_ret != NOTFOUND) {
						switch (del_ret)
						{
						case NORMAL:
							float *mbr;
							mbr = succ -> get_mbr();
							memcpy(entries[i].bounces, mbr, sizeof(float) * 2 * dimension);
							dirty = true;
							delete [] mbr;
							entries[i].del_son();
							return NORMAL;
							break;
						case ERASED:
							entries[i].del_son();
							int j;
							for (j = i; j < num_entries - 1; j++)
								entries[j] = entries[j+1];
							for (j = num_entries - 1; j < capacity; j++)
								entries[j].son_ptr = NULL;

							num_entries--;
							dirty = true;
							succ = NULL;

							if (num_entries < (int)ceil(0.4 * capacity)) {
								for (int j = 0; j < num_entries; j++) {
									Entry* e = new Entry(dimension,NULL);
									*e = entries[j];
									my_tree -> deletelist.push_front(e);
								}
								my_tree -> num_of_inodes --;
								return ERASED;
							} else
								return NORMAL;
							break;
						}
					}
				}
			}
		}
	} else {
		for (int i = 0; i < num_entries; i++) {
			if (entries[i] == (*e)) {
				my_tree -> num_of_data --;

				for (int j = i; j < num_entries-1; j++)
					entries[j] = entries[j+1];

				num_entries--;
				dirty = true;
				if (this != my_tree -> root_ptr && num_entries < (int)ceil(0.4 * capacity))
				{
					for (int k = 0; k < num_entries; k++) {
						Entry* en = new Entry(dimension,NULL);

						*en = entries[k];
						en -> level = 0;
						my_tree -> deletelist.push_front(en);
					}
					my_tree -> num_of_dnodes --;
					return ERASED;
				}
				else
					return NORMAL;
			}
		}
		return NOTFOUND;
	}
	return NORMAL;		// dummy return, avoid warning
}

void RTNode::enter(Entry *de) {
  	//note that de will be deleted after being entered.
    if (num_entries > (capacity-1))
        error("RTNode::enter: called, but node is full", true);
    entries[num_entries] = *de;
    num_entries++;
	dirty = true;
    de->son_ptr = NULL;
    delete de;	// why this line leads to seg. fault ?
}

bool RTNode::FindLeaf(Entry *e) {
	RTNode *succ;
	if (level > 0) {
		for (int i = 0; i < num_entries; i++) {
			float *f= overlapRect(my_tree -> dimension,
				  		entries[i].bounces, e -> bounces);
			if (f != NULL) {
				delete [] f;
				succ = entries[i].get_son();
				bool find;
				find = succ->FindLeaf(e);
				entries[i].del_son();
				if (find)
					return true;
			}
		}
		return false;
	} else {
		for (int i = 0; i < num_entries; i++) {
			if (entries[i] == (*e)) return true;
		}
		return false;
	}
	return false;
}

float* RTNode::get_mbr() {
    float *mbr= new float[2*dimension];
    for (int i = 0; i < 2*dimension; i ++ )
        mbr[i] = entries[0].bounces[i];

    for (int j = 1; j < num_entries; j++) {
    	for (int i = 0; i < 2*dimension; i += 2) {
    	    mbr[i]   = min(mbr[i],   entries[j].bounces[i]);
    	    mbr[i+1] = max(mbr[i+1], entries[j].bounces[i+1]);
        }
    }
    return mbr;
}

R_OVERFLOW RTNode::insert(Entry *d, RTNode **sn) {
    int follow;
    RTNode *succ, *new_succ;
    RTNode *brother;
    Entry *de;
    R_OVERFLOW ret;
    float *mbr,*nmbr;

    int i, last_cand;
    float *center;
    SortMbr *sm;
    Entry *new_entries;

    if (level > 0) { // direcrtory node
	  if (level > d -> level) {
        follow = choose_subtree(d -> bounces);
        succ = entries[follow].get_son();
        ret = succ -> insert(d, &new_succ);
        mbr = succ -> get_mbr();
        memcpy(entries[follow].bounces, mbr, sizeof(float) * 2 * dimension);
        delete [] mbr;

		entries[follow].del_son();

        if (ret == SPLIT) {
        	// node has split into itself and *new_succ
            if (num_entries == capacity)
         	    error("RTNode::insert: maximum capacity violation", true);

            de = new Entry(dimension, my_tree);
    	    nmbr = new_succ -> get_mbr();
            memcpy(de -> bounces, nmbr, 2 * dimension * sizeof(float));
    	    delete [] nmbr;
            de -> son = new_succ -> block;
			delete new_succ;
            de -> son_ptr = NULL;
            enter(de);

            if (num_entries == (capacity - 1)) {
        	    brother = new RTNode(my_tree);
        	    my_tree -> num_of_inodes++;
        	    brother -> level = level;
        	    split(brother);
                *sn = brother;
                ret = SPLIT;
        	} else
          	    ret = NONE;
        }
        dirty = true;
        return ret;
	  } else {//level==d->level
		  enter(d);    //note that d will be deleted on return
		  if (num_entries == (capacity - 1)) {
            // maximun no of entries --> Split
            // this happens already if the node is nearly filled
            // for the algorithms are more easy then
        	brother = new RTNode(my_tree);
        	my_tree -> num_of_inodes++;
        	brother -> level = level;
        	split(brother);
            *sn = brother;
            ret = SPLIT;
		  } else
          	ret = NONE;

		  dirty=true;
		  return ret;
	  }
    } else { // data (leaf) node
        if (num_entries == capacity)
        	error("RTDataNode::insert: maximum capacity violation", true);

        enter(d);
        dirty = true;

        if (num_entries == (capacity - 1)) {
	        // maximum # of entries --> Split
	        // this happens already if the node is nearly filled
	        // for the algorithms are more easy then

            if (my_tree->re_level[0] == false && my_tree -> root_ptr -> level != level) {
	    	    // there was no reinsert on level 0 during this insertion
				//Here I changed the condition as if it is already root, no need
				//to reinsert.  Split directly in this case  -----------By TAO Yufei

                // calculate center of page
                mbr = get_mbr();
                center = new float[dimension];
                for (i = 0; i < dimension; i++)
                     center[i] = (mbr[2*i] + mbr[2*i+1]) / 2.0f;
                new_entries = new Entry[capacity];

				for (i = 0; i < capacity; i ++)
					new_entries[i].init_entry(dimension, my_tree);

        	    sm = new SortMbr[num_entries];
        	    for (i = 0; i < num_entries; i++) {
            		sm[i].index = i;
            		sm[i].dimension = dimension;
            		sm[i].mbr = entries[i].bounces;
            		sm[i].center = center;
                }

                qsort(sm, num_entries, sizeof(SortMbr), sort_center_mbr);

                last_cand = (int) ((float)num_entries * 0.30);

                // copy the nearest candidates to new array
                for (i = 0; i < num_entries - last_cand; i++)
    	            new_entries[i] = entries[sm[i].index];

                // insert candidates into reinsertion list
                for ( ; i < num_entries; i++) {
                	Entry* nd = new Entry(dimension,NULL);
                	*nd = entries[sm[i].index];
                    my_tree -> re_data_cands .push_front(nd);
                }

                // free and copy data array
                delete [] entries;
        	    entries = new_entries;

        	    delete sm;
        	    delete [] mbr;
        	    delete [] center;
        	    my_tree -> re_level[0] = true;
        	    num_entries -= last_cand;	// correct # of entries
        	    dirty = true;	// must write page
                return REINSERT;
        	} else {  //there has been reinsertion on this level
        	    *sn = new RTNode(my_tree);
        	    (*sn) -> level = level;
        	    my_tree -> num_of_dnodes++;
        	    split((RTNode *) *sn);
    	    }
    	    return SPLIT;
        } else
            return NONE;
    }
}

void RTNode::update_count() {
	dirty=true;
	for (int i = 0; i < num_entries ; i++) {
		if (is_data_node()) {
			entries[i].num_data=1;
		} else {
			RTNode *rtn=entries[i].get_son();
			rtn->update_count();
			entries[i].num_data=0;
			for (int j=0;j<rtn->num_entries;j++)
				entries[i].num_data+=rtn->entries[j].num_data;
			entries[i].del_son();
			//printf("%d %d %d\n",level,i,entries[i].num_data);
		}
    }
}

void RTNode::print() {
	printf("level %d  Block: %d\n", level, block);
    for (int i = 0; i < num_entries ; i++) {
        printf("(%4.1lf, %4.1lf, %4.1lf, %4.1lf)\n",
	       entries[i].bounces[0],
	       entries[i].bounces[1],
	       entries[i].bounces[2],
	       entries[i].bounces[3]);
    }
}

void RTNode::read_from_buffer(char *buffer) {
    int i, j;
    // Level
    memcpy(&level, buffer, sizeof(char));
    j = sizeof(char);

    // num_entries
    memcpy(&num_entries, &(buffer[j]), sizeof(int));
    j += sizeof(int);

    int s = entries[0].get_size();
    for (i = 0; i < num_entries; i++) {
    	entries[i].read_from_buffer(&buffer[j]);
    	j += s;
    }
}

int RTNode::split(float **mbr, int **distribution) {
    bool lu;
    int i, j, k, l, s, n, m1, dist, split_axis;
    SortMbr *sml, *smu;
    float minmarg, marg, minover, mindead, dead, over, *rxmbr, *rymbr;

    n = num_entries;
    m1 = (int) ceil((float)n * 0.40);
    sml = new SortMbr[n];
    smu = new SortMbr[n];
    rxmbr = new float[2*dimension];
    rymbr = new float[2*dimension];

    // choose split axis
    minmarg = MAXREAL;
    for (i = 0; i < dimension; i++) {	// for each axis
        for (j = 0; j < n; j++) {
            sml[j].index = smu[j].index = j;
            sml[j].dimension = smu[j].dimension = i;
            sml[j].mbr = smu[j].mbr = mbr[j];
        }

        // Sort by lower and upper value perpendicular axis_i
      	qsort(sml, n, sizeof(SortMbr), sort_lower_mbr);
        qsort(smu, n, sizeof(SortMbr), sort_upper_mbr);

        marg = 0.0;
        // for all possible distributions of sml
        for (k = 0; k < n - 2 * m1 + 1; k++) {
			for (s = 0; s < 2 * dimension; s += 2) {
				rxmbr[s] =    MAXREAL;
				rxmbr[s+1] = -MAXREAL;
			}
            for (l = 0; l < m1 + k; l++) {
				for (s = 0; s < 2*dimension; s += 2) {
					rxmbr[s] =   min(rxmbr[s],   sml[l].mbr[s]);
					rxmbr[s+1] = max(rxmbr[s+1], sml[l].mbr[s+1]);
				}
			}
			marg += margin(dimension, rxmbr);

			for (s = 0; s < 2 * dimension; s += 2) {
				rxmbr[s] =    MAXREAL;
				rxmbr[s+1] = -MAXREAL;
			}
            for ( ; l < n; l++) {
				for (s = 0; s < 2 * dimension; s += 2) {
					rxmbr[s] =   min(rxmbr[s],   sml[l].mbr[s]);
					rxmbr[s+1] = max(rxmbr[s+1], sml[l].mbr[s+1]);
				}
            }
			marg += margin(dimension, rxmbr);
        }

        // for all possible distributions of smu
       	for (k = 0; k < n - 2 * m1 + 1; k++) {
            // now calculate margin of R1
			// initialize mbr of R1
			for (s = 0; s < 2 * dimension; s += 2) {
				rxmbr[s] =    MAXREAL;
				rxmbr[s+1] = -MAXREAL;
			}
            for (l = 0; l < m1+k; l++) {
                // calculate mbr of R1
				for (s = 0; s < 2 * dimension; s += 2) {
					rxmbr[s] =   min(rxmbr[s],   smu[l].mbr[s]);
					rxmbr[s+1] = max(rxmbr[s+1], smu[l].mbr[s+1]);
				}
            }
			marg += margin(dimension, rxmbr);

            // now calculate margin of R2
			// initialize mbr of R2
			for (s = 0; s < 2 * dimension; s += 2) {
				rxmbr[s] =    MAXREAL;
				rxmbr[s+1] = -MAXREAL;
			}
            for ( ; l < n; l++) {
                // calculate mbr of R1
				for (s = 0; s < 2 * dimension; s += 2) {
					rxmbr[s] =   min(rxmbr[s],   smu[l].mbr[s]);
					rxmbr[s+1] = max(rxmbr[s+1], smu[l].mbr[s+1]);
				}
            }
			marg += margin(dimension, rxmbr);
        }

        if (marg < minmarg) {
            split_axis = i;
            minmarg = marg;
        }
    }

    // choose best distribution for split axis
    for (j = 0; j < n; j++) {
		sml[j].index = smu[j].index = j;
		sml[j].dimension = smu[j].dimension = split_axis;
		sml[j].mbr = smu[j].mbr = mbr[j];
    }

    // Sort by lower and upper value perpendicular split axis
    qsort(sml, n, sizeof(SortMbr), sort_lower_mbr);
    qsort(smu, n, sizeof(SortMbr), sort_upper_mbr);

    minover = MAXREAL;
    mindead = MAXREAL;
    // for all possible distributions of sml and snu
    for (k = 0; k < n - 2 * m1 + 1; k++) {
        dead = 0.0;
		for (s = 0; s < 2 * dimension; s += 2) {
			rxmbr[s] =    MAXREAL;
			rxmbr[s+1] = -MAXREAL;
		}
		for (l = 0; l < m1 + k; l++) {
			for (s = 0; s < 2*dimension; s += 2) {
				rxmbr[s] =   min(rxmbr[s],   sml[l].mbr[s]);
				rxmbr[s+1] = max(rxmbr[s+1], sml[l].mbr[s+1]);
			}
			dead -= area(dimension, sml[l].mbr);
		}
        dead += area(dimension, rxmbr);
		//**************note**************
		//this does not compute the dead space for all the cases.  some overlapping
		//area may be subtrated twice.

		for (s = 0; s < 2 * dimension; s += 2) {
			rymbr[s] =    MAXREAL;
       		rymbr[s+1] = -MAXREAL;
		}
		for ( ; l < n; l++) {
			for (s = 0; s < 2*dimension; s += 2) {
				rymbr[s] =   min(rymbr[s],   sml[l].mbr[s]);
				rymbr[s+1] = max(rymbr[s+1], sml[l].mbr[s+1]);
			}
			dead -= area(dimension, sml[l].mbr);
		}
        dead += area(dimension, rymbr);

		over = overlap(dimension, rxmbr, rymbr);

        if ((over < minover) || (over == minover) && dead < mindead) {
            minover = over;
            mindead = dead;
            dist = m1+k;
            lu = true;
        }

		//Now we do the same thing for smu
        dead = 0.0;
		for (s = 0; s < 2*dimension; s += 2) {
			rxmbr[s] =    MAXREAL;
			rxmbr[s+1] = -MAXREAL;
		}
		for (l = 0; l < m1+k; l++) {
			for (s = 0; s < 2*dimension; s += 2) {
				rxmbr[s] =   min(rxmbr[s],   smu[l].mbr[s]);
				rxmbr[s+1] = max(rxmbr[s+1], smu[l].mbr[s+1]);
			}
			dead -= area(dimension, smu[l].mbr);
		}
        dead += area(dimension, rxmbr);

		for (s = 0; s < 2*dimension; s += 2) {
			rymbr[s] =    MAXREAL;
			rymbr[s+1] = -MAXREAL;
		}
		for ( ; l < n; l++) {
			for (s = 0; s < 2*dimension; s += 2) {
				rymbr[s] =   min(rymbr[s],   smu[l].mbr[s]);
				rymbr[s+1] = max(rymbr[s+1], smu[l].mbr[s+1]);
			}
			dead -= area(dimension, smu[l].mbr);
		}
		//correcting errors
        dead += area(dimension, rymbr);
		over = overlap(dimension, rxmbr, rymbr);
        if ((over < minover) || (over == minover) && dead < mindead) {
            minover = over;
            mindead = dead;
            dist = m1+k;
            lu = false;
        }
    }

    // calculate best distribution
	// the array distribution is deleted in split(rtnode *sn);
    *distribution = new int[n];
    for (i = 0; i < n; i++) {
        if (lu)
            (*distribution)[i] = sml[i].index;
        else
            (*distribution)[i] = smu[i].index;
    }
    delete [] sml;
    delete [] smu;
    delete [] rxmbr;
    delete [] rymbr;
    return dist;
}

void RTNode::split(RTNode *sn) {
    int i, *distribution, dist, n;
    float **mbr_array;
    Entry *new_entries1, *new_entries2;

    n = num_entries;
    mbr_array = new float*[n];
    for (i = 0; i < n; i++)
       	mbr_array[i] = entries[i].bounces;

    dist = split(mbr_array, &distribution);
    new_entries1 = new Entry[capacity];
    new_entries2 = new Entry[capacity];
	for (i = 0; i < capacity; i ++) {
		new_entries1[i].init_entry(dimension, my_tree);
		new_entries2[i].init_entry(dimension, my_tree);
	}

    for (i = 0; i < dist; i++)
       	new_entries1[i] = entries[distribution[i]];

    for (i = dist; i < n; i++)
       	new_entries2[i-dist] = entries[distribution[i]];

    for (i = 0; i < n; i++) {
       	entries[i].son_ptr = NULL;
       	sn->entries[i].son_ptr = NULL;
    }
    delete [] entries;
    delete [] sn->entries;

    entries = new_entries1;
    sn->entries = new_entries2;

    num_entries = dist;
    sn->num_entries = n - dist;

    delete [] mbr_array;
	delete [] distribution;
}

void RTNode::write_to_buffer(char *buffer) {
    int i, j, s;

    // Level
    memcpy(buffer, &level, sizeof(char));
    j = sizeof(char);

    // num_entries
    memcpy(&buffer[j], &num_entries, sizeof(int));
    j += sizeof(int);

    s = entries[0].get_size();
    for (i = 0; i < num_entries; i++) {
    	entries[i].write_to_buffer(&buffer[j]);
       	j += s;
    }
}

RTree::RTree(char *fname, int _b_length, Cache *c, int _dimension) {
  	//use this constructor to build a new tree
  	leaf_acc=non_leaf_acc=0;
    file = new BlockFile(fname, _b_length);
    cache = c;

    dimension = _dimension;
    root = 0;
    root_ptr = NULL;
    root_is_data = true;
    num_of_data = num_of_inodes = num_of_dnodes = 0;

    root_ptr = new RTNode(this);
	  //note that when a tree is constructed, the root is automatically created
	  //though at this time there is no entry in the root yet.
    num_of_dnodes++;
    root_ptr -> level = 0;
    root = root_ptr -> block;
}

RTree::RTree(char *fname,float cache_factor) {
  	//use this constructor to restore a tree from a file
    leaf_acc=non_leaf_acc=0;
    file = new BlockFile(fname, 0);
    cache = new Cache((int)(cache_factor*file->get_num_of_blocks()),file->blocklength);

    char *header = new char [file->get_blocklength()];
    file -> read_header(header);
    read_header(header);
	delete [] header;
    root_ptr = NULL;
}

RTree::~RTree() {
	char *header = new char[file -> get_blocklength()];
    write_header(header);
    file->set_header(header);
    delete [] header;
	del_root();
	if (cache) {
		cache->flush();
		delete cache;	// cache destroyed here ...
	}
	if (file->new_flag)
	    printf("This R-Tree contains %d internal, %d data nodes and %d data\n",
		   		num_of_inodes, num_of_dnodes, num_of_data);
    delete file;
}

void RTree::del_root() {
	if (root_ptr!=NULL) {
		delete root_ptr;
		root_ptr = NULL;
	}
}

bool RTree::delete_entry(Entry *d) {
	load_root();

	R_DELETE del_ret;
	del_ret=root_ptr->delete_entry(d);

	if (del_ret == NOTFOUND) return false;
	if (del_ret == ERASED)
		error("RTree::delete_entry--The root has been deleted\n",true);

	if (root_ptr -> level > 0 && root_ptr -> num_entries == 1) {
		//there is only one entry in the root but the root
		//is not leaf.  in this case, the child of the root is exhalted to root
		root = root_ptr -> entries[0].son;
		del_root();
		load_root();
		num_of_inodes--;
	}

	//Now will reinsert the entries
    /*
	while (deletelist.size()> 0) {
		Entry *new_e=(Entry*) deletelist.front();
		deletelist.pop_front();
		insert(new_e);
	}*/
	del_root();
	return true;
}

void RTree::insert(Entry* d) {
    int i, j;
    RTNode *sn;
    RTNode *nroot_ptr;
    int nroot;
    Entry *de;
    R_OVERFLOW split_root;
    Entry *dc;
    float *nmbr;

    // load root into memory
    load_root();

    // no overflow occured until now
    re_level = new bool[root_ptr -> level + 1];
    for (i = 0; i <= root_ptr -> level; i++) re_level[i] = false;

    // insert d into re_data_cands as the first entry to insert
    // make a copy of d because it should be erased later
    Entry* new_link = new Entry(dimension,NULL);
	*new_link = *d;
	re_data_cands.push_front(new_link);

	delete d;  //we follow the convention that the entry will be deleted when insertion finishes

    j = -1;
    while (re_data_cands.size()>0) {
        // first try to insert data, then directory entries
	    Entry *dc=(Entry*)re_data_cands.front();
        if (dc != NULL) {
            // since "erase" deletes the data itself from the
            // list, we should make a copy of the data before
            // erasing it
			re_data_cands.pop_front();

            // start recursive insert with root
			split_root = root_ptr -> insert(dc, &sn);
        } else
	        error("RTree::insert: inconsistent list re_data_cands", true);

    	if (split_root == SPLIT) {
    		// insert has lead to split --> new root-page with two sons (i.e. root and sn)
    	    nroot_ptr = new RTNode(this);
    	    nroot_ptr -> level = root_ptr -> level + 1;
    	    num_of_inodes++;
    	    nroot = nroot_ptr -> block;

    	    de = new Entry(dimension, this);
    	    nmbr = root_ptr -> get_mbr();
    	    memcpy(de->bounces, nmbr, 2*dimension*sizeof(float));
    	    delete [] nmbr;
    	    de->son = root_ptr->block;
    	    de->son_ptr = root_ptr;
    	    nroot_ptr -> enter(de);

    	    de = new Entry(dimension, this);
    	    nmbr = sn -> get_mbr();
    	    memcpy(de -> bounces, nmbr, 2*dimension*sizeof(float));
    	    delete [] nmbr;
    	    de -> son = sn -> block;
    	    de -> son_ptr = sn;
    	    nroot_ptr->enter(de);

    	    root = nroot;
            root_ptr = nroot_ptr;
            root_is_data = false;
        }
        j++;
    }
    num_of_data++;
    delete [] re_level;
	del_root();
}

void RTree::load_root() {
	if (root_ptr == NULL)
        root_ptr = new RTNode(this, root);
}

void RTree::read_header(char *buffer) {
    int i;
    memcpy(&dimension, buffer, sizeof(dimension));
    i = sizeof(dimension);

    memcpy(&num_of_data, &buffer[i], sizeof(num_of_data));
    i += sizeof(num_of_data);

    memcpy(&num_of_dnodes, &buffer[i], sizeof(num_of_dnodes));
    i += sizeof(num_of_dnodes);

    memcpy(&num_of_inodes, &buffer[i], sizeof(num_of_inodes));
    i += sizeof(num_of_inodes);

    memcpy(&root_is_data, &buffer[i], sizeof(root_is_data));
    i += sizeof(root_is_data);

    memcpy(&root, &buffer[i], sizeof(root));
    i += sizeof(root);
}

void RTree::write_header(char *buffer) {
    int i;
    memcpy(buffer, &dimension, sizeof(dimension));
    i = sizeof(dimension);

    memcpy(&buffer[i], &num_of_data, sizeof(num_of_data));
    i += sizeof(num_of_data);

    memcpy(&buffer[i], &num_of_dnodes, sizeof(num_of_dnodes));
    i += sizeof(num_of_dnodes);

    memcpy(&buffer[i], &num_of_inodes, sizeof(num_of_inodes));
    i += sizeof(num_of_inodes);

    memcpy(&buffer[i], &root_is_data, sizeof(root_is_data));
    i += sizeof(root_is_data);

    memcpy(&buffer[i], &root, sizeof(root));
    i += sizeof(root);
}


