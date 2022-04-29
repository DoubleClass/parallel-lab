#include<iostream>
#include<sys/time.h>
#include<arm_neon.h>
using namespace std;

const int MAXN=2048;
int N;
const double eps=1e-3;

float a[MAXN][MAXN];
float b[MAXN][MAXN];

char cases[6][26]={"./data/mat_dim64.txt","./data/mat_dim128.txt","./data/mat_dim256.txt","./data/mat_dim512.txt","./data/mat_dim1024.txt","./data/mat_dim2048.txt"};
int range[6]={64,128,256,512,1024,2048};





class CTimer
{
public:
	CTimer(void);
	~CTimer(void);

	void time_in();
	long long time_out();

private:
	struct timeval start;
	struct timeval end;
};

CTimer::CTimer(void)
{
}


CTimer::~CTimer(void)
{
}

void CTimer::time_in()
{
	gettimeofday(&start,NULL);
}

long long CTimer::time_out()
{
	gettimeofday(&end,NULL);
	long long timer=1000000 * (end.tv_sec-start.tv_sec)+ end.tv_usec-start.tv_usec;
	return timer;
}



void read(){
	for(int i=0;i<N;i++)
		for(int j=0;j<N;j++){
			cin>>a[i][j];
		}
	for(int i=0;i<N;i++)
		for(int j=0;j<N;j++){
			cin>>b[i][j];
		}
}

void generate_data(int n) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			a[i][j] = rand()%100;
			b[i][j] = rand()%100;
		}
	}
}

class Trivial{
public:
	float A[MAXN][MAXN];
	float res[MAXN][MAXN];
	long long time_used;
	void read(){
		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++){
				A[i][j]=a[i][j];
			}
		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++){
				res[i][j]=b[i][j];
			}
	}
	void calculate(){
		time_used = 0;
		CTimer ct;
		ct.time_in();
		for(int k=0;k<N;k++){
			for(int j=k+1;j<N;j++){
				A[k][j]=A[k][j]/A[k][k];
			}
			A[k][k]=1.0f;
			for(int i=k+1;i<N;i++){
				for(int j=k+1;j<N;j++){
					A[i][j]=A[i][j]-A[i][k]*A[k][j];
				}
				A[i][k]=0;
			}
		}
		time_used = ct.time_out();
		printf("    trivial algo costs:%lldms    ", time_used);
	}
	void check(){
		bool flag=0;
		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++){
				if(abs(A[i][j]-res[i][j])>eps)flag=1;
			}
		if(flag){
			cout<<"wrong"<<endl;
		}
		else{
			cout<<"correct"<<endl;
		}
	}

}trivial_algo;


class SIMD{
public:
	float A[MAXN][MAXN];
	float res[MAXN][MAXN];
	long long time_used;
	void read(){
		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++){
				A[i][j]=a[i][j];
			}
		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++){
				res[i][j]=b[i][j];
			}
	}
	void calculate(){
		time_used = 0;
		CTimer ct;
		ct.time_in();
		for(int k=0;k<N;k++){
			float32x4_t vl= vld1q_dup_f32(&A[k][k]);
			for(int j=k+1;j+4<=N;j+=4){
				float32x4_t va = vld1q_f32(&A[k][j]);
				va=vdivq_f32(va,vl);
				vst1q_f32(&A[k][j],va);
			}
			A[k][k]=1.0f;
			for(int i=k+1;i<N;i++){
				float32x4_t vaik= vld1q_dup_f32(&A[i][k]);
				for(int j=k+1;j+4<=N;j+=4){
					float32x4_t vakj = vld1q_f32(&A[k][j]);
					float32x4_t vaij = vld1q_f32(&A[i][j]);
					float32x4_t vx=vmulq_f32(vakj,vaik);
					vaij=vsubq_f32(vaij,vx);
					vst1q_f32(&A[i][j],vaij);
				}
				A[i][k]=0;
			}
		}
		time_used = ct.time_out();

		printf("   SIMD algo costs:%lldms    ",time_used);
	}
	void check(){
		bool flag=0;
		for(int i=0;i<N;i++)
			for(int j=0;j<N;j++){
				if(abs(A[i][j]-res[i][j])>eps){if(flag==0)cout<<A[i][j]<<" "<<res[i][j];flag=1;}
			}
		if(flag){
			cout<<"wrong"<<endl;
		}
		else{
			cout<<"correct"<<endl;
		}
	}
}SIMD_algo;


int main(){
	for(int i=0;i<6;i++){
		N=range[i];
		string prefix = "./data/";
		freopen(cases[i],"r",stdin);
		cout<<"current dimension: "<< N <<endl;
		//generate_data(2048);
		read();
		trivial_algo.read();
		trivial_algo.calculate();
		trivial_algo.check();
		SIMD_algo.read();
		SIMD_algo.calculate();
		SIMD_algo.check();

		long long t1 = trivial_algo.time_used;
		long long t2 = SIMD_algo.time_used;
		cout << "t1 is " << "**********" << t1 << endl;
		cout << "t2 is " << "**********" << t2 << endl;

		
        cout<<"impored: "<<(t1-t2)/t1*100<<"%"<<endl<<endl;
	}
}