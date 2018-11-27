#include <array>
#include <cassert>
#include <cstdio>
#include <omp.h>

using namespace std;

size_t CUTOFF = 4;

size_t serial_count_inversions(int *data, size_t length) {
    /* Sort data array in ascending order and count inversions */
    size_t count = 0;

    if (length <= 1) {
        assert(length == 1);
        return count;
    }

    // Count inversions in left and right halves
    size_t half = length / 2;
    int *data_prime = data + half;
    size_t length_prime = length - half;
    
    size_t left = serial_count_inversions(data, half);
    size_t right = serial_count_inversions(data_prime, length_prime);
    count = left + right;
    
    // Merge sort two halves and count inversions
    int *temp = new int[length];
    size_t i, l = 0, r = 0;

    for (i = 0; i < length; ++i) {
        if (l == half) {
            temp[i] = data_prime[r];
            r++;
        } else if (r == length_prime) {
            temp[i] = data[l];
            l++;
        } else {
            if (data_prime[r] < data[l]) {
                temp[i] = data_prime[r];
                count += half - l;
                r++;
            } else {
                temp[i] = data[l];
                l++;
            }
        }
    }

    // Copy data to original tab
    for (i = 0; i < length; ++i) {
        data[i] = temp[i];
    }

    delete[] temp;
    return count;
}

size_t parallel_count_inversions(int *data, size_t length, bool in_parallel=true) {
    /* Sort data array in ascending order and count inversions */
    if (length <= CUTOFF) {
        return serial_count_inversions(data, length);
    }
    
    size_t count = 0;

    // Count inversions in left and right halves
    size_t half = length / 2;
    int *data_prime = data + half;
    size_t length_prime = length - half;

#pragma omp parallel shared(count) if(in_parallel)
#pragma omp single nowait
    {
        size_t left, right;
#pragma omp task shared(left)
        left = parallel_count_inversions(data, half, false);
#pragma omp task shared(right)
        right = parallel_count_inversions(data_prime, length_prime, false);
#pragma omp taskwait
        count = left + right;
    }

    // Merge sort two halves and count inversions
    int *temp = new int[length];
    size_t i, l = 0, r = 0;

    for (i = 0; i < length; ++i) {
        if (l == half) {
            temp[i] = data_prime[r];
            r++;
        } else if (r == length_prime) {
            temp[i] = data[l];
            l++;
        } else {
            if (data_prime[r] < data[l]) {
                temp[i] = data_prime[r];
                count += half - l;
                r++;
            } else {
                temp[i] = data[l];
                l++;
            }
        }
    }

    // Copy data to original tab
    for (i = 0; i < length; ++i) {
        data[i] = temp[i];
    }

    delete[] temp;
    return count;
}

int main(int argc, char **argv) {
    array<int, 18> test = {-1, -4, -3, 6, 1, 7, 2, 9, 3, 0, 5, 4, 15, 11, 12, 10, 13, 14};

    printf("unsorted array: ");
    for (int i = 0; i < test.size(); ++i)
        printf("%d, ", test[i]);
    printf("\n");
    
    size_t count = parallel_count_inversions(test.data(), test.size());

    printf("  sorted array: ");
    for (int i = 0; i < test.size(); ++i)
        printf("%d, ", test[i]);
    printf("\ncount inversions: %d\n", count);

    assert(count == 28);
    
    return 0;
}
