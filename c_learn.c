#include <stdio.h>

int main() {
    printf("Hello, World from C!\\n");
    return 0;
} 


// New Code



#include <stdio.h>
#include <string.h>

void reverseArray(char* arr) {
    int n = 0;
    while (arr[n] != '\0') {
        n++;
    }
    // int n = strlen(arr);  #this can find the length of arr using string.h
    int temp;
    int i;

    for (i = 0; i < n / 2; i++) {
        temp = arr[i];
        arr[i] = arr[n - 1 - i];
        arr[n - 1 - i] = temp;
    }
}

int main() {
    char arr[] = "Hello World";

    printf("Original: %s\n", arr);
    reverseArray(arr);
    printf("Reversed: %s\n", arr);

    return 0;
}


// compile: - gcc c_learn.c -o hello
// run    - ./hello
        