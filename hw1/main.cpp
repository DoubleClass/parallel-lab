#include <iostream>
#include <stdio.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <windows.h>
#include <time.h>
#include <math.h>
using namespace std;
long long head, tail, freq;
void print_matrix(float**A, int n);
float* normal_LU(float** A, int n, float *b) {
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int k = 0; k < n; k++) {
        float diagonal = A[k][k];
        for (int j = k + 1; j < n; j++) {
            A[k][j] = A[k][j] / diagonal;
        }
        b[k] = b[k]/diagonal;
        A[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];

            }
            b[i] = b[i] - A[i][k] * b[k];
            A[i][k] = 0;
        }
    }

    float *x = new float[n];
    x[n - 1] = b[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        float sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum);
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "normal LU of size " << n << " cost: " << (tail - head) * 1000.0 / freq << "ms" << endl;
    return x;
}

float* SSE_LU_1(float **A, int n, float *b) { // work with eliminatino
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    __m128 t1, t2, t3, t4;
    for (int k = 0; k < n; k++) {
        float tmp[4] = { A[k][k], A[k][k], A[k][k], A[k][k] };
        t1 = _mm_loadu_ps(tmp);
        for (int j = n - 4; j >= k; j -= 4) {
            t2 = _mm_loadu_ps(A[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(A[k] + j, t3);
        }

        if (k % 4 != (n % 4)) {
            for (int j = k; j % 4 != ( n% 4); j++){
                A[k][j] = A[k][j] / tmp[0];
            }
        }
//        for (int j = (n % 4) - 1; j>= 0; j--) {
//            A[k][j] = A[k][j] / tmp[0];
//        }
        b[k] = b[k]/tmp[0];
        //A[k][k] = 1.0;
        //print_matrix(A, n);
        for (int i = k + 1; i < n; i++) {
            float tmp[4] = { A[i][k], A[i][k], A[i][k], A[i][k] };
            t1 = _mm_loadu_ps(tmp);
            for (int j = n - 4; j > k;j -= 4) {
                t2 = _mm_loadu_ps(A[i] + j);
                t3 = _mm_loadu_ps(A[k] + j);
                t4 = _mm_sub_ps(t2,_mm_mul_ps(t1, t3)); //����
                _mm_storeu_ps(A[i] + j, t4);

            }
            for (int j = k + 1; j % 4 !=(n % 4); j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            b[i] = b[i] - A[i][k] * b[k];
            A[i][k] = 0;
        }
        //print_matrix(A, n);
    }
    float *x = new float[n];
    x[n - 1] = b[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        float sum = 0;
        for (int j = i + 1; j < n; j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum);
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "SSE 1 of size " << n << " cost: " << (tail - head) * 1000.0 / freq << "ms" << endl;
    return x;

}

float* SSE_LU_2(float **A, int n, float *b) { // work with back substitution
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    __m128 t1, t2, t3, t4;
    for (int k = 0; k < n; k++) {
        float diagonal = A[k][k];
        for (int j = k + 1; j < n; j++) {
            A[k][j] = A[k][j] / diagonal;
        }
        b[k] = b[k]/diagonal;
        A[k][k] = 1.0;
        for (int i = k + 1; i < n; i++) {
            for (int j = k + 1; j < n; j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];

            }
            b[i] = b[i] - A[i][k] * b[k];
            A[i][k] = 0;
        }
    }

    float *x = new float[n];
    x[n - 1] = b[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        float tmp[4] = {0.0, 0.0, 0.0, 0.0};
        t1 = _mm_load_ps(tmp);
        for (int j = n - 4; j > i; j -= 4) {
            t2 = _mm_loadu_ps(x + j);
            t3 = _mm_loadu_ps(A[i] + j);
            t1 = _mm_add_ps(t1, _mm_mul_ps(t3, t2));


        }

        _mm_storeu_ps(tmp, t1);
        float sum = tmp[0] + tmp[1] + tmp [2] + tmp[3];
        for (int j = i + 1; j % 4 != (n % 4); j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum);
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "SSE 2 of size " << n << " cost: " << (tail - head) * 1000.0 / freq << "ms" << endl;
    return x;

}

float* SSE_LU_both(float **A, int n, float *b) {
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    __m128 t1, t2, t3, t4;
    for (int k = 0; k < n; k++) {
        float tmp[4] = { A[k][k], A[k][k], A[k][k], A[k][k] };
        t1 = _mm_loadu_ps(tmp);
        for (int j = n - 4; j >= k; j -= 4) {
            t2 = _mm_loadu_ps(A[k] + j);
            t3 = _mm_div_ps(t2, t1);
            _mm_storeu_ps(A[k] + j, t3);
        }

        if (k % 4 != (n % 4)) {
            for (int j = k; j % 4 != ( n% 4); j++){
                A[k][j] = A[k][j] / tmp[0];
            }
        }
//        for (int j = (n % 4) - 1; j>= 0; j--) {
//            A[k][j] = A[k][j] / tmp[0];
//        }
        b[k] = b[k]/tmp[0];
        //A[k][k] = 1.0;
        //print_matrix(A, n);
        for (int i = k + 1; i < n; i++) {
            float tmp[4] = { A[i][k], A[i][k], A[i][k], A[i][k] };
            t1 = _mm_loadu_ps(tmp);
            for (int j = n - 4; j > k;j -= 4) {
                t2 = _mm_loadu_ps(A[i] + j);
                t3 = _mm_loadu_ps(A[k] + j);
                t4 = _mm_sub_ps(t2,_mm_mul_ps(t1, t3)); //����
                _mm_storeu_ps(A[i] + j, t4);

            }
            for (int j = k + 1; j % 4 !=(n % 4); j++) {
                A[i][j] = A[i][j] - A[i][k] * A[k][j];
            }
            b[i] = b[i] - A[i][k] * b[k];
            A[i][k] = 0;
        }
        //print_matrix(A, n);
    }
    float *x = new float[n];
    x[n - 1] = b[n - 1];
    for (int i = n - 2; i >= 0; i--) {
        float tmp[4] = {0.0, 0.0, 0.0, 0.0};
        t1 = _mm_load_ps(tmp);
        for (int j = n - 4; j > i; j -= 4) {
            t2 = _mm_loadu_ps(x + j);
            t3 = _mm_loadu_ps(A[i] + j);
            t1 = _mm_add_ps(t1, _mm_mul_ps(t3, t2));


        }

        _mm_storeu_ps(tmp, t1);
        float sum = tmp[0] + tmp[1] + tmp [2] + tmp[3];
        for (int j = i + 1; j % 4 != (n % 4); j++) {
            sum += A[i][j] * x[j];
        }
        x[i] = (b[i] - sum);
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << "SSE 1 and 2 of size " << n << " cost: " << (tail - head) * 1000.0 / freq << "ms" << endl;
    return x;

}

void print_matrix(float** A, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%8lf ", A[i][j]);
        }
        cout << endl;
    }
    cout << endl;
}

int main()
{
    int N;
//    cin >> N;

    for (int m = 4; m < 13; m++) {
        N = power(2, m);
        srand((unsigned)time(NULL));
        float **A = new float*[N];
        for (int i = 0; i < N; i++) {
            A[i] = new float[N]();
        }
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                cin >>A[i][j];
                //A[i][j] = rand() % 100;
            }
        }
        float *b1 = new float[N];
        float *b2 = new float[N];

        for (int i = 0; i < N; i++) {
            //b1[i] = rand()%100;
        cin >> b1[i];
        }

    //    float *x1 =  normal_LU(A, N, b1);
    //    print_matrix(A, N);

        float *x2 = SSE_LU_both(A,N,b1);
        for (int i = 0; i < N; i++){
            cout << x2[i] << " ";
            cout <<endl;
        }
    }

//    srand((unsigned)time(NULL));
//    float **A = new float*[N];
//    for (int i = 0; i < N; i++) {
//        A[i] = new float[N]();
//    }
//    for (int i = 0; i < N; i++) {
//        for (int j = 0; j < N; j++) {
//            cin >>A[i][j];
//            //A[i][j] = rand() % 100;
//        }
//    }
//    float *b1 = new float[N];
//    float *b2 = new float[N];
//
//    for (int i = 0; i < N; i++) {
////        b1[i] = rand()%100;
//          cin >> b1[i];
//    }
//
////    float *x1 =  normal_LU(A, N, b1);
////    print_matrix(A, N);
//
//    float *x2 = SSE_LU_BP(A,N,b1);
//    //float *x2 = normal_LU(A,N,b1);
//    //print_matrix(A, N);
//
//    for (int i = 0; i < N; i++){
//        cout << x2[i] << " " <<endl;
//    }
//    cout <<endl;
//    return 0;
}
