
#include <iostream>
#include <math.h>
#include <cstdlib>
#include <stdlib.h>
#include <fstream>
#include <pthread.h>
#include <omp.h>
#define THREAD_NUM 4

using namespace std;

typedef struct {
    int threadId;
}threadParm_t;

pthread_mutex_t mutex;

float A[10240][10240];
float B[10240][10240];

int n = 512;

void print_matric(int n, float matrix[][10240]) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

void init_0(int n, float matrix[][10240]) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < i; j++) {
			matrix[i][j] = 0;
		}
		matrix[i][i] = 1;
	}
}

void init_rand() {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			A[i][j] = i + j + 1 + rand() * 100;
			B[i][j] = A[i][j];
		}
	}
}

void normal_LU(int n) {
	for (int k = 0; k < n; k++) {
		for (int j = k + 1; j < n; j++) {
			A[k][j] = A[k][j] / A[k][k];
		}
		A[k][k] = 1.0;
		for (int i = k + 1; i < n; i++) {
			for (int l = k + 1; l<n; l++) {
				A[i][l] = A[i][l] - A[i][k] * A[k][l];
			}
			A[i][k] = 0;
		}
	}
}





void LU_openmp(int n) {

//    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
//    QueryPerformanceCounter((LARGE_INTEGER *)&head);
	for (int k = 0; k < n; k++) {
        # pragma omp parallel for num_threads(THREAD_NUM)
		for (int j = k + 1; j < n; j++) {
			A[k][j] = A[k][j] / A[k][k];
		}
		A[k][k] = 1.0;
		# pragma omp parallel for num_threads(THREAD_NUM)
		for (int i = k + 1; i < n; i++) {
			for (int l = k + 1; l<n; l++) {
				A[i][l] = A[i][l] - A[i][k] * A[k][l];
			}
			A[i][k] = 0;
		}
	}
//	pthread_mutex_lock(&mutex);
//
//	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
//	cout << "omp cost:  " << (tail - head) * 1000.0 / freq << "ms" << endl;
//	pthread_mutex_unlock(&mutex);
}

// void LU_openmp_SSE(int n) {
//     long long tail;
// 	__m128 t1, t2, t3;

// 	for (int k = 0; k < n;k++) {
//         //cout << "k is " << k << endl;
//         # pragma omp parallel for num_threads(THREAD_NUM)
// 		for (int i = k + 1; i < n; i++) {
//             //printf("i is %d, n is %d\n", i, n);
//             B[i][k] = B[i][k] / B[k][k];
//             //printf("%lf\n", B[i][k]);
//             int offset = (n - k - 1) % 4;

//             for (int j = k + 1; j < k+1+offset; j++) {
//                 B[i][j] = B[i][j] - B[i][k] * B[k][j];
//             }
//             t2 = _mm_set_ps(B[i][k], B[i][k], B[i][k], B[i][k]);

//             for (int j = k + 1 + offset ; j<n; j+=4) {
//                 t3 = _mm_load_ps(B[k] + j);
//                 t1 = _mm_load_ps(B[i] + j);
//                 t2 = _mm_mul_ps(t2, t3);
//                 t1 = _mm_sub_ps(t1, t2);
//                 _mm_store_ps(B[i] + j, t1);
//             }


// 		}
// 	}

// 	pthread_mutex_lock(&mutex);

// 	QueryPerformanceCounter((LARGE_INTEGER *)&tail);
// 	cout << "openMP with SSE " << (tail - head) * 1000.0 / freq << "ms" << endl;
// 	pthread_mutex_unlock(&mutex);

// }


int main() {
	double time = 0;
	long long tail;
	int i;
	//QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
	n = 1000;
	cout << "�����ģ:" << n << endl;
	init_rand();

    /***********��ͨLU�㷨******************/
	//QueryPerformanceCounter((LARGE_INTEGER *)&head);
	normal_LU(n);
	//QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	// time = (tail - head) * 1000.0 / freq;
	// cout << "single thread: " << time << "ms" << endl;
    /***********openMP �㷨*****************/
    //QueryPerformanceCounter((LARGE_INTEGER *)&head);
    // for (int i = 1 ; i < 100; i++) {

    // }
	LU_openmp(n);
	//QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	// time = (tail - head) * 1000.0 / freq;
	// cout << "openmp: " << time << "ms" << endl;

	/***********openMP SSE �㷨*****************/
	// init_rand();
    // QueryPerformanceCounter((LARGE_INTEGER *)&head);
	// LU_openmp_SSE(n);
	// //QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	// //time = (tail - head) * 1000.0 / freq;
	// //cout << "openmp: " << time << "ms" << endl;

	// //thread in block
	// /***********�Ӿ����зֵ�pthread*****************/
	// init_rand();
	// cout << "cut in matrices" << endl;
	// mutex = PTHREAD_MUTEX_INITIALIZER;
	// threadParm_t threadParm[THREAD_NUM];
	// pthread_t thread[THREAD_NUM];
	// pthread_barrier_init(&barrier, NULL, THREAD_NUM);

	// QueryPerformanceCounter((LARGE_INTEGER*)&head);
	// for (i = 0; i < THREAD_NUM; i++) {
	// 	threadParm[i].threadId = i;
	// 	pthread_create(&thread[i], NULL, LU_pthread, (void *)&threadParm[i]);
	// }

	// for (int i = 0; i < THREAD_NUM; i++) {
	// 	pthread_join(thread[i], 0);
	// }
    // cout << endl;
    // /***********��ѭ���зֵ�pthread*****************/
    // init_rand();
    // cout << "cut in loop" << endl;
	// pthread_mutex_destroy(&mutex);

    // //thread in loop
	// mutex = PTHREAD_MUTEX_INITIALIZER;
	// pthread_barrier_init(&barrier, NULL, THREAD_NUM);

	// QueryPerformanceCounter((LARGE_INTEGER*)&head);
	// for (i = 0; i < THREAD_NUM; i++) {
	// 	threadParm[i].threadId = i;
	// 	pthread_create(&thread[i], NULL, LU_pthread_matrix, (void *)&threadParm[i]);
	// }

	// for (int i = 0; i < THREAD_NUM; i++) {
	// 	pthread_join(thread[i], 0);
	// }

	// pthread_mutex_destroy(&mutex);
	// cout << endl;

    // /***********��ѭ���зֵ�pthread + SSE*****************/
	// cout << "cut in loop with sse" << endl;
	// init_rand();
	// mutex = PTHREAD_MUTEX_INITIALIZER;
	// pthread_barrier_init(&barrier, NULL, THREAD_NUM);
	// QueryPerformanceCounter((LARGE_INTEGER*)&head);
	// for (i = 0; i < THREAD_NUM; i++) {
	// 	threadParm[i].threadId = i;
	// 	pthread_create(&thread[i], NULL, LU_pthread_sse, (void *)&threadParm[i]);
	// }

	// for (int i = 0; i < THREAD_NUM; i++) {
	// 	pthread_join(thread[i], 0);
	// }
    // QueryPerformanceCounter((LARGE_INTEGER *)&tail);
	// time = (tail - head) * 1000.0 / freq;
	// cout << "sse: " << time << "ms" << endl;

	// pthread_mutex_destroy(&mutex);

	system("pause");
}

