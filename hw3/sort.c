#include <stdio.h>
#include <time.h>
#include <mpi.h>
#include <stdlib.h>
const int ARR_NUM = 10000;
const int ARR_LEN = 10000;
int seg = 2500;
int arr[10000][10000];

int comp(const void*a, const void*b) {
	return *(int*)a-*(int*)b;
}

void init_unbalanced() {
    int ratio;
    srand(time(0));
    for (int i = 0; i < ARR_NUM; i++) {
        
        if (i < seg) ratio = 0;
        else if (i < seg * 2) ratio = 32;
        else if (i < seg * 3) ratio = 64;
        else ratio = 128;
        if ((rand() & 127) < ratio)
            for (int j = 0; j < ARR_LEN; j++)
                arr[i][j] = ARR_LEN - j;
        else
            for (int j = 0; j < ARR_LEN; j++)
                arr[i][j] = j;
    }
}

void normal_sort() {
    for (int i = 0; i < ARR_NUM; i++) {
        qsort(arr[i], 10000, sizeof(int), comp);
    }
}

void printArr(int n) {
    for (int i = 0; i < 10; i++) {
        printf("%d ", arr[n][i]);
    }
    printf("\n");
}

int main() {
//     init_unbalanced();
//     for (int i = 1; i < 40; i++) {
// 	    printf("%d ", arr[6000][i]);
// 	}
// 	qsort(arr[6000], 10000, sizeof(int), comp);
// 	printf("\nnext\n");
// 	for (int i = 1; i < 40; i++) {
// 	    printf("%d ", arr[6000][i]);
// 	}
    printf("inside main");
    init_unbalanced();
//     for (int i = 1; i < 4000; i++) {
// 	    printf("%d ", arr[6000][i]);
// 	}
    int my_rank, comm_sz, local_n, start;
    
    MPI_Init(NULL, NULL);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
	printf("local \n");
	local_n = ARR_NUM / comm_sz;
    start = my_rank * local_n;
	for (int i = start; i < start + local_n; i++) {
	    qsort(arr[i], 10000, sizeof(int), comp);
	    
	}
	printf("exec sort from %d to %d, and rank is %d\n", start, start + local_n - 1, my_rank);
	printArr(6000);
	if (my_rank == 0) {
	    printf("this  is rank 0\n");
	} else {
	    printf("this is rank %d\n", my_rank);
	}
	MPI_Finalize();
	printArr(6000);
	return 0;
}





