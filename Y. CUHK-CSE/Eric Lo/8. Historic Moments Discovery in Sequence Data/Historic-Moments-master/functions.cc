#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "gendef.h"



/////////////////////////  functions for convex polygon operations

void AddVertex(FloatVec& vec,tPoint a) {
	int vlen=vec.size();
	if (vec.size()==0) { // first point
		vec.push_back(a[0]);
		vec.push_back(a[1]);
	} else if (!((a[0]==vec[vlen-2]&&a[1]==vec[vlen-1])||(a[0]==vec[0]&&a[1]==vec[1]))) {
		// add vertex if it is not first point or previous point
		vec.push_back(a[0]);
		vec.push_back(a[1]);
	}
}

int	AreaSign( tPoint a, tPoint b, tPoint c ) {
    float area2 = (b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]);

    // 0.00001 is a very small number (tends to 0)
    if      ( area2 >  0.00001 ) return  1;
    else if ( area2 < -0.00001 ) return -1;
    else                     return  0;
}

// Advances and prints out an inside vertex if appropriate.
inline int Advance(FloatVec& vec, int a, int *aa, int n, bool inside, tPoint v ) {
	if ( inside )
		AddVertex(vec,v);
	(*aa)++;
	return  (a+1) % n;
}

// Returns true iff point c lies on the closed segement ab.
// Assumes it is already known that abc are collinear.
bool Between( tPoint a, tPoint b, tPoint c ) {
   /* If ab not vertical, check betweenness on x; else on y. */
   if ( a[0] != b[0] )
      return ((a[0] <= c[0]) && (c[0] <= b[0])) || ((a[0] >= c[0]) && (c[0] >= b[0]));
   else
      return ((a[1] <= c[1]) && (c[1] <= b[1])) || ((a[1] >= c[1]) && (c[1] >= b[1]));
}

void Assignd(tPoint p, tPoint a, tPoint q, tPoint b) {
	for (int i = 0; i < 2; i++ ) {
		p[i] = a[i];
		q[i] = b[i];
	}
}

char ParallelInt( tPoint a, tPoint b, tPoint c, tPoint d, tPoint p, tPoint q) {
   if ( AreaSign( a, b, c ) != 0 )	// not collinear
      return '0';

   if ( Between( a, b, c ) && Between( a, b, d ) ) {
      Assignd( p, c, q, d );
      return 'e';
   }
   if ( Between( c, d, a ) && Between( c, d, b ) ) {
      Assignd( p, a, q, b );
      return 'e';
   }
   if ( Between( a, b, c ) && Between( c, d, b ) ) {
      Assignd( p, c, q, b );
      return 'e';
   }
   if ( Between( a, b, c ) && Between( c, d, a ) ) {
      Assignd( p, c, q, a );
      return 'e';
   }
   if ( Between( a, b, d ) && Between( c, d, b ) ) {
      Assignd( p, d, q, b );
      return 'e';
   }
   if ( Between( a, b, d ) && Between( c, d, a ) ) {
      Assignd( p, d, q, a );
      return 'e';
   }
   return '0';
}


/*SegSegInt: Finds the point of intersection p between two closed
segments ab and cd.  Returns p and a char with the following meaning:
   'e': The segments collinearly overlap, sharing a point.
   'v': An endpoint (vertex) of one segment is on the other segment,
        but 'e' doesn't hold.
   '1': The segments intersect properly (i.e., they share a point and
        neither 'v' nor 'e' holds).
   '0': The segments do not intersect (i.e., they share no points).
Note that two collinear segments that share just one point, an endpoint
of each, returns 'e' rather than 'v' as one might expect. */
char SegSegInt( tPoint a, tPoint b, tPoint c, tPoint d, tPoint p, tPoint q ) {
	float  s, t;       /* The two parameters of the parametric eqns. */
	float num, denom;  /* Numerator and denoninator of equations. */
 	char code = '?';    /* Return char characterizing intersection. */

	denom=a[0]*(d[1]-c[1]) + b[0]*(c[1]-d[1]) + d[0]*(b[1]-a[1]) + c[0]*(a[1]-b[1]);

	/* If denom is zero, then segments are parallel: handle separately. */
	if (denom == 0.0)
		return  ParallelInt(a, b, c, d, p, q);

	num=a[0]*(d[1]-c[1]) + c[0]*(a[1]-d[1]) + d[0]*(c[1]- a[1]);
	if ( (num == 0.0) || (num == denom) )
		code = 'v';
	s = num / denom;

	num=-(a[0]*(c[1]-b[1]) + b[0]*(a[1]-c[1]) + c[0]*(b[1]-a[1]) );
	if ( (num == 0.0) || (num == denom) )
		code = 'v';
	t = num / denom;

	if ( (0.0 < s) && (s < 1.0) && (0.0 < t) && (t < 1.0) )
		code = '1';
	else if ( (0.0 > s) || (s > 1.0) || (0.0 > t) || (t > 1.0) )
		code = '0';

	p[0] = a[0] + s * ( b[0] - a[0] );
	p[1] = a[1] + s * ( b[1] - a[1] );
	return code;
}

// Returns the dot product of the two input vectors
float  Dot( tPoint a, tPoint b ) {
    float sum = 0.0;
    for (int i = 0; i < 2; i++ )
       sum += a[i] * b[i];
    return  sum;
}

// a - b ==> c
void SubVec( tPoint a, tPoint b, tPoint c ) {
   for (int i = 0; i < 2; i++ )
      c[i] = a[i] - b[i];
}


bool InPoly(tPoint q,int n,tPoint* PX) {
	int	 i, i1;      /* point index; i1 = i-1 mod n */
	float  x;          /* x intersection of e with ray */
	int	 Rcross = 0; /* number of right edge/ray crossings */
	int  Lcross = 0; /* number of left edge/ray crossings */

	/* Shift so that q is the origin. Note this destroys the polygon.*/
	tPoint P[n];
	for( i = 0; i < n; i++ ) {
		P[i][0] = PX[i][0] - q[0];
		P[i][1] = PX[i][1] - q[1];
	}

	/* For each edge e=(i-1,i), see if crosses ray. */
	for (i = 0; i < n; i++ ) {
		if ( P[i][0]==0 && P[i][1]==0 )
			return true;
		i1 = ( i + n - 1 ) % n;

		if( ( P[i][1] > 0 ) != ( P[i1][1] > 0 ) ) {
			x = (P[i][0] * P[i1][1] - P[i1][0] * P[i][1]) / (P[i1][1] - P[i][1]);
			if (x > 0) Rcross++;
		}

		if ( ( P[i][1] < 0 ) != ( P[i1][1] < 0 ) ) {
			x = (P[i][0] * P[i1][1] - P[i1][0] * P[i][1]) / (P[i1][1] - P[i][1]);
			if (x < 0) Lcross++;
		}
	}

	/* q on the edge if left and right cross are not the same parity. */
	if( ( Rcross % 2 ) != (Lcross % 2 ) )
		return true;

	/* q inside iff an odd number of crossings. */
	if( (Rcross % 2) == 1 )
		return true;
	else
		return false;
}

// seems to be necessary
bool CheckPolyCover(FloatVec& vec,int n,tPoint* P, int m,tPoint* Q) {
	// if all vertices in another polygon , then ...

	int match_count=0;
	for (int i=0;i<n;i++) {
		if (InPoly(P[i],m,Q)) {
			vec.push_back(P[i][0]);
			vec.push_back(P[i][1]);
			match_count++;
		} else
			break;
	}
	if (match_count==n)
		return true;
	else
		vec.clear();


	match_count=0;
	for (int i=0;i<m;i++) {
		if (InPoly(Q[i],n,P)) {
			vec.push_back(Q[i][0]);
			vec.push_back(Q[i][1]);
			match_count++;
		} else
			break;
	}
	if (match_count==m)
		return true;
	else
		vec.clear();

	return false;
}




// P has n vertices, Q has m vertices
bool PolyPolyIntersect(FloatVec& vec, int n, float* P_ptr, int m, float* Q_ptr) {
	int     a, b;           /* indices on P and Q (resp.) */
	int     a1, b1;         /* a-1, b-1 (resp.) */
	tPoint A, B;           /* directed edges on P and Q (resp.) */
	int     cross;          /* sign of z-component of A x B */
	int     bHA, aHB;       /* b in H(A); a in H(b). */
	tPoint Origin = {0,0}; /* (0,0) */
	tPoint p;              /* float point of intersection */
	tPoint q;              /* second point of intersection */
	tInFlag inflag;         /* {Pin, Qin, Unknown}: which inside */
	int     aa, ba;         /* # advances on a & b indices (after 1st inter.) */
	bool    FirstPoint;     /* Is this the first point? (used to initialize).*/
	tPoint p0;             /* The first point. */
	int     code;           /* SegSegInt return code. */

	a = 0; b = 0; aa = 0; ba = 0;
	inflag = Unknown; FirstPoint = true;
	tPoint* P=(tPoint*)P_ptr;
	tPoint* Q=(tPoint*)Q_ptr;
	vec.clear();


	// handle case where one is completely inside the other
	if (CheckPolyCover(vec,n,P,m,Q))
		return true;
	else
		vec.clear();


	do {
		// Computations of key variables.
		a1 = (a + n - 1) % n;
		b1 = (b + m - 1) % m;

		SubVec( P[a], P[a1], A );
		SubVec( Q[b], Q[b1], B );

		cross = AreaSign( Origin, A, B );
		aHB   = AreaSign( Q[b1], Q[b], P[a] );
		bHA   = AreaSign( P[a1], P[a], Q[b] );

		// If A & B intersect, update inflag.
		code = SegSegInt( P[a1], P[a], Q[b1], Q[b], p, q );
		if ( code == '1' || code == 'v' ) {
			if ( inflag == Unknown && FirstPoint ) {
				aa = ba = 0;
				FirstPoint = false;
				p0[0] = p[0]; p0[1] = p[1];
				AddVertex(vec,p0);	// move-to pt.
			}
			AddVertex(vec,p);

			// Update inflag
			if (aHB > 0)
				inflag = Pin;
			else if ( bHA > 0)
				inflag = Qin;
		}

		// Special case: A & B overlap and oppositely oriented.
		if ( ( code == 'e' ) && (Dot( A, B ) < 0) ) {
			//AddVertex(vec,p);	// moveto
			//AddVertex(vec,q);	// lineto
			return false;	// false for opposite oriented
		}

		// Special case: A & B parallel and separated.
		if ( (cross == 0) && ( aHB < 0) && ( bHA < 0 ) ) {	// 	P and Q are disjoint
			return false;
		} else if ( (cross == 0) && ( aHB == 0) && ( bHA == 0 ) ) { 	// Special case: A & B collinear.
			// Advance but do not output point.
			if ( inflag == Pin )
				b = Advance(vec, b, &ba, m, inflag == Qin, Q[b] );
			else
				a = Advance(vec, a, &aa, n, inflag == Pin, P[a] );
		}
		// Generic cases.
		else if (cross >= 0 ) {
			if ( bHA > 0)
				a = Advance(vec, a, &aa, n, inflag == Pin, P[a] );
			else
				b = Advance(vec, b, &ba, m, inflag == Qin, Q[b] );
		} else { // i.e.,  cross < 0
			if ( aHB > 0)
				b = Advance(vec, b, &ba, m, inflag == Qin, Q[b] );
			else
				a = Advance(vec, a, &aa, n, inflag == Pin, P[a] );
		}
	// Quit when both adv. indices have cycled, or one has cycled twice
	} while ( ((aa < n) || (ba < m)) && (aa < 2*n) && (ba < 2*m) );

	// Deal with special cases: not implemented
	if ( inflag == Unknown) { // boundaries of P and Q do not cross
		return false;
	}

	return true;
}

bool PolyMbrIntersect(FloatVec& vec, int n, float* P_ptr, float* bounces) {
	float Q_ptr[8];	// 4 points, each with 2 coordinates

	Q_ptr[0]=bounces[0];	Q_ptr[1]=bounces[2];	// low-left
	Q_ptr[2]=bounces[1];	Q_ptr[3]=bounces[2];	// low-right
	Q_ptr[4]=bounces[1];	Q_ptr[5]=bounces[3];	// high-right
	Q_ptr[6]=bounces[0];	Q_ptr[7]=bounces[3];	// high-left

	return  PolyPolyIntersect( vec, n, P_ptr, 4, Q_ptr);
}

// Reads a polygon from stdin, returns the number of vertices
/*int ReadPoly(float* P) { // assume enough space
	int nin;

	scanf("%d", &nin);
	for (int i=0;i<nin;i++)
		scanf("%f %f",&P[2*i],&P[2*i+1]);

	return  nin;
}*/

/*void PrintPoly(int n, float* P) {
	printf("Polygon intersection:\n");
	printf("i\tx\ty\n");
	for (int i = 0; i < n; i++ )
		printf("%d\t%f\t%f\n", i, P[2*i], P[2*i+1]);
}*/

void PrintPoly(FloatVec& vec) {
	int len=vec.size()/2;
	printf("Polygon intersection:\n");
	printf("i\tx\ty\n");
	for (int i = 0; i < len; i++ )
		printf("%d\t%f\t%f\n", i, vec[2*i], vec[2*i+1]);
}

// requirement: all points with respect to origin
int sort_anticlock(const void *d1, const void *d2) {
    float *s1=(float *) d1, *s2=(float *) d2;
    double diff=s2[0]*s1[1]-s1[0]*s2[1];

    // ascending of slope
    if (diff<0)
        return -1;
    else if (diff>0)
        return 1;
    else {	// ascending of y
    	 if (s1[1]<s2[1])
	        return -1;
	    else if (s1[1]>s2[1])
	        return 1;
	    else
    		return 0;
    }
}


void AntiClockwiseSort(int n,float* P_ptr) {
	float x_0=P_ptr[0],y_0=P_ptr[1];
	int slot=0;

	for (int i=0;i<n;i++) {	// find min. x
		if (P_ptr[2*i]<x_0) {
			x_0=P_ptr[2*i];
			y_0=P_ptr[2*i+1];
			slot=i;
		}
	}
	// swap element at "slot" with the last one
	P_ptr[2*slot]=P_ptr[2*n-2];		P_ptr[2*slot+1]=P_ptr[2*n-1];
	P_ptr[2*n-2]=x_0;				P_ptr[2*n-1]=y_0;

	for (int i=0;i<n;i++) {
		P_ptr[2*i]-=x_0;	P_ptr[2*i+1]-=y_0;
	}

	qsort(P_ptr,n-1,2*sizeof(float),sort_anticlock);	// sort only first n-1 points

	for (int i=0;i<n;i++) {
		P_ptr[2*i]+=x_0;	P_ptr[2*i+1]+=y_0;
	}
}

/////////////////////////


void error(char *t, bool ex) {
    fprintf(stderr, t);
    if (ex) exit(0);
}

float area(int dimension, float *mbr) {
    float sum = 1.0;
    for (int i = 0; i < dimension; i++)
		sum *= mbr[2*i+1] - mbr[2*i];
    return sum;
}

float margin(int dimension, float *mbr) {
    float *ml, *mu, *m_last, sum;
    sum = 0.0;
    m_last = mbr + 2*dimension;
    ml = mbr;
    mu = ml + 1;
    while (mu < m_last) {
		sum += *mu - *ml;
		ml += 2;
		mu += 2;
    }
    return sum;
}

bool inside(float &p, float &lb, float &ub) {
    return (p >= lb && p <= ub);
}

// calcutales the overlapping rectangle between r1 and r2
// if rects do not overlap returns null
float* overlapRect(int dimension, float *r1, float *r2) {
	float *overlap = new float[2*dimension];
	for (int i=0; i<dimension; i++) {
	    if ((r1[i*2]>r2[i*2+1]) || (r1[i*2+1]<r2[i*2])) { // non overlapping
	        delete [] overlap;
			return NULL;
		}
		overlap[2*i] = max(r1[i*2], r2[i*2]);
	    overlap[2*i+1] = min(r1[i*2+1], r2[i*2+1]);
	}
	return overlap;
}

float overlap(int dimension, float *r1, float *r2) {
	// calcutales the overlapping area of r1 and r2
	// calculate overlap in every dimension and multiplicate the values
    float *r1pos, *r2pos, *r1last, r1_lb, r1_ub, r2_lb, r2_ub;
    float sum = 1.0;
    r1pos = r1; r2pos = r2;
    r1last = r1 + 2 * dimension;
    while (r1pos < r1last) {
		r1_lb = *(r1pos++);
		r1_ub = *(r1pos++);
		r2_lb = *(r2pos++);
		r2_ub = *(r2pos++);
        // calculate overlap in this dimension

        if (inside(r1_ub, r2_lb, r2_ub)) {
        	// upper bound of r1 is inside r2
            if (inside(r1_lb, r2_lb, r2_ub))
            	// and lower bound of r1 is inside
                sum *= (r1_ub - r1_lb);
            else
                sum *= (r1_ub - r2_lb);
		} else {
            if (inside(r1_lb, r2_lb, r2_ub))
	    	// and lower bound of r1 is inside
				sum *= (r2_ub - r1_lb);
	    	else {
				if (inside(r2_lb, r1_lb, r1_ub)&&inside(r2_ub, r1_lb, r1_ub))
	        		sum *= (r2_ub - r2_lb);		// r1 contains r2
				else
					sum = 0.0;					// r1 and r2 do not overlap
	    	}
		}
    }
    return sum;
}

void enlarge(int dimension, float **mbr, float *r1, float *r2) {
	// enlarge r in a way that it contains s
    *mbr = new float[2*dimension];
    for (int i = 0; i < 2*dimension; i += 2) {
		(*mbr)[i]   = min(r1[i],   r2[i]);
		(*mbr)[i+1] = max(r1[i+1], r2[i+1]);
    }
}

void print_mbr(float *mbr,char* msg) {
	assert(DIMENSION==2);
	printf("[%f %f] [%f %f] ",mbr[0],mbr[0],mbr[2],mbr[3]);
	if (msg==NULL)
		printf("\n");
	else
		printf("%s\n",msg);
}

int sort_lower_mbr(const void *d1, const void *d2) {
    SortMbr *s1, *s2;
    s1 = (SortMbr *) d1;
    s2 = (SortMbr *) d2;
    int dimension = s1->dimension;
    float erg = s1->mbr[2*dimension] - s2->mbr[2*dimension];
    if (erg < 0.0)
		return -1;
    else if (erg == 0.0)
		return 0;
    else
		return 1;
}

int sort_upper_mbr(const void *d1, const void *d2) {
    SortMbr *s1, *s2;

    s1 = (SortMbr *) d1;
    s2 = (SortMbr *) d2;
    int dimension = s1->dimension;
    float erg = s1->mbr[2*dimension+1] - s2->mbr[2*dimension+1];
    if (erg < 0.0)
		return -1;
    else if (erg == 0.0)
		return 0;
    else
		return 1;
}

int sort_center_mbr(const void *d1, const void *d2) {
    SortMbr *s1, *s2;
    float d, e1, e2;

    s1 = (SortMbr *) d1;
    s2 = (SortMbr *) d2;
    int dimension = s1->dimension;

    e1 = e2 = 0.0;
    for (int i = 0; i < dimension; i++) {
        d = ((s1->mbr[2*i] + s1->mbr[2*i+1]) / 2.0f) - s1->center[i];
        e1 += d*d;
        d = ((s2->mbr[2*i] + s2->mbr[2*i+1]) / 2.0f) - s2->center[i];
        e2 += d*d;
    }

    if (e1 < e2)
		return -1;
    else if (e1 == e2)
		return 0;
    else
		return 1;
}



bool MBR_section(float *mbr,float* bounces) {
    bool overlap=true;
    for (int i = 0; i < DIMENSION; i++) {
		if (mbr[2 * i] > bounces[2 * i + 1] ||  mbr[2 * i + 1] < bounces[2 * i])
			overlap = false;
    }
    return overlap;
}

float MINDIST_SQR(float *bounces1, float *bounces2) {
	float r,summe=0.0;
	for (int i = 0; i < DIMENSION; i++) {
		r=0;
		if (bounces1[2*i+1]<bounces2[2*i])
			r=bounces2[2*i]-bounces1[2*i+1];
		else if (bounces2[2*i+1]<bounces1[2*i])
			r=bounces1[2*i]-bounces2[2*i+1];
		summe += r*r;
	}
    return summe;
}

float MAXDIST_SQR(float *bounces1, float *bounces2) {
    float r,summe=0.0;
    for (int i = 0; i < DIMENSION; i++) {
    	r=max( bounces2[2*i+1]-bounces1[2*i] , bounces1[2*i+1]-bounces2[2*i] );
		summe += r*r;
    }
    return summe;
}


float MINDIST(float *bounces1, float *bounces2) {
    float r,summe=0.0;
    for (int i = 0; i < DIMENSION; i++) {
    	r=0;
		if (bounces1[2*i+1]<bounces2[2*i])
			r=bounces2[2*i]-bounces1[2*i+1];
		else if (bounces2[2*i+1]<bounces1[2*i])
			r=bounces1[2*i]-bounces2[2*i+1];
		summe += r*r;
    }
    return sqrt(summe);
}

float MAXDIST(float *bounces1, float *bounces2) {
    float r,summe=0.0;
    for (int i = 0; i < DIMENSION; i++) {
    	r=max( bounces2[2*i+1]-bounces1[2*i] , bounces1[2*i+1]-bounces2[2*i] );
		summe += r*r;
    }
    return sqrt(summe);
}

// only applicable to 2D data
void getPolyMBR(float* bounces,FloatVec& vec) {
	bounces[0]=bounces[1]=bounces[2]=bounces[3]=0;	// for the case len==0

	int len=vec.size()/2;
	for (int m=0;m<len;m++) {	// check each corner
		float curx=vec[2*m],cury=vec[2*m+1];

		if (m==0) {
			bounces[0]=bounces[1]=curx;
			bounces[2]=bounces[3]=cury;
		} else {
			bounces[0]=min(bounces[0],curx);
			bounces[1]=max(bounces[1],curx);
			bounces[2]=min(bounces[2],cury);
			bounces[3]=max(bounces[3],cury);
		}
	}
}

float MINMAXDIST(float *_p, float *bounces) {
    // Nearest Narbor Query v. Roussopoulos, Kelley und Vincent,

    float summe = 0;
    float minimum = 1.0e20f;
    float S = 0;
    float p[DIMENSION];


    float rmk, rMi;
    int k,i;

    for( i = 0; i < DIMENSION; i++) {
		p[i]=_p[2*i];	// wrapping

		rMi = (	p[i] >= (bounces[2*i]+bounces[2*i+1])/2 )
			? bounces[2*i] : bounces[2*i+1];
		S += float((p[i] - rMi)*(p[i] - rMi));
    }

    for( k = 0; k < DIMENSION; k++)
    {
		rmk = ( p[k] <=  (bounces[2*k]+bounces[2*k+1]) / 2 ) ?
			bounces[2*k] : bounces[2*k+1];
		summe = float((p[k] - rmk)*(p[k] - rmk));
		rMi = (	p[k] >= (bounces[2*k]+bounces[2*k+1]) / 2 )
			? bounces[2*k] : bounces[2*k+1];
		summe += S - float((p[k] - rMi)*(p[k] - rMi));
		minimum = min( minimum,summe);
    }
    return(minimum);
}


