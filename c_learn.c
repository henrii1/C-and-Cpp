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


//NEW CODE

#include <stdio.h>

int main() {

    long nc;

    nc = 0;
    while(getchar() != EOF)
    ++nc;
    printf("%ld\n", nc);
}

// counts the number of chars in input. will require EOF from input (ctrl + Z + enter) in windows
// ++nc means pre-increment. the value is incremented before being used. the other is post increment.


//NEW CODE

//Replace tabs with appropriate spaces

//loop through input till EOF. If tab calculate the space and append for that amount else append c

#include <stdio.h>

#define TAB_SIZE 8

void detab(FILE *input, FILE *output, int tab_size){
    int position = 0;
    int c;  //fgetc returns int value of characters. fputc takes the number and writes the char.
    
    while ((c=fgetc(input)) != EOF){
        if (c == '\t'){
            int spaces = tab_size - (position % tab_size);

            for (int i=0; i<spaces; i++){
                fputc(' ', output);

            }
            position++;
        } else {
            fputc(c, output);
        }
        if (c == '\n')
            position = 0;
    }

}

int main(){
    printf("Enter text. press (CTRL + Z + Enter to terminate on Windows:\n");
    detab(stdin, stdout, TAB_SIZE);
    return 0;
}


//NEW CODE
//break line into multiline after tab or blank 1-22

//loop through input till EOF. If tab calculate the space and append for that amount else append c

#include <stdio.h>

#define MAX_COL 80

void fold(FILE *input, FILE *output){
    int c;  //fgetc returns int value of characters. fputc takes the number and writes the char.
    int col= 0;
    int last_blank = -1;
    char line[MAX_COL + 1];
    int i = 0;

    while((c=fgetc(input) != EOF)){
        if (c == '\n'){
            line[i] = '\0';
            fputs(line, output);
            fputc('\n', output);
            col = 0;
            i = 0;
            last_blank = -1;
        } else{
            line[i] = c;
            i++;
            col++;
            if (c == ' ' || '\t'){
                last_blank = i;
            }

        if (col >= MAX_COL){
            if (last_blank != -1){
                line[last_blank] = '\0';
                fputs(line, output);
                fputc('\n', output);

                // for the other half

                for (int j = last_blank + 1; j<i; j++){
                    line[j-last_blank-1] = line[j];
                }
                i -= (last_blank + 1);
                col = i;
                last_blank = -1;
            } else {
                line[i] = '\0';
                fputs(line, output);
                fputc('\n', output);
                
                i = 0;
                col = 0;
                last_blank= -1;
            }
        }
        }

    }
            // remaining characters
            if (i > 0 ){
                line[i] = '\0';
                fputs(line, output);
            }
}


int main(){
    printf("Enter text. press (CTRL + Z + Enter to terminate on Windows:\n");
    fold(stdin, stdout);
    return 0;
}




// compile: - gcc c_learn.c -o hello
// run    - ./hello
        

//NEW CODE


// write a code to remove all comments from c code.
#include <stdio.h>
#include <stdlib.h>

#define MAX_BUFFER 1024

void strip(FILE *input, FILE *output){
 
     char c;
     int in_string = 0, in_char = 0;
     int prev_char = 0;

     while((c=fgetc(input)) != EOF){
        if (in_string) {
            if (c == '"' && prev_char != '\\'){
                in_string = 0;
            }
            fputc(c, output);
        } else if (in_char) {
            if (c == '\'' && prev_char != '\\'){
                in_char = 0;
            }
            fputc(c, output);
        } else {
            if (c == '/' && prev_char == '/'){
                while((c=fgetc(input)) != EOF && c != '\n'){
                    if (c != EOF){
                        fputc('\n', output);
                    }
                    prev_char = 0;
                    continue;
                } else if (c == '*' && prev_char == '/'){
                    while ((c=fgetc(input)) != EOF) {
                        if (c == '/' && prev_char =='*'){
                            break;
                        }
                        prev_char = c;
                    }
                    prev_char = 0;
                    continue;
                } else {
                    if (c == '"'){
                        in_string = 1;
                    }
                    if (prev_char) {
                        fputc(prev_char, output);
                    }
                }
            }
            prev_char = c;
        }
        if (prev_char) {
            fputc(prev_char, output);
        }
     }


int main(int argc, char *argv[]) {
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input file> <output file> \n", argv[0])
        return 1;
    }

    FILE *input = fopen(argv[1], "r");
    if (!input){
        perror("Failed to open input file");
        return 1;
    }

    FILE *output = fopen(argv[2], "w");
    if (!output) {
        perror("Failed to open output file");
        fclose(input);
        return 1;
    }

    strip(input, output)

    fclose(input);
    fclose(output);

    return 0;
}


//NEW CODE

//create a function that convert an integer 'n' into a base 'b' representation in the string 's'

//we'll use the division method
#include <stdio.h>
#include <string.h>

void reverse(char s[]) {
    int i, j;
    char c;
    for (i=0, j=strlen(s)-1; i < j; i++, j--){
        c = s[i];
        s[i] = s[j];
        s[j] = c;
    }
}

void itob(int n, char s[], int b) {
    int i = 0, sign;
    int digit;
    //convert negative value to positive
    if((sign = n) < 0){
        n = -n;
    }
    do {
        digit = n % b;
        s[i++] = (digit>9) ? (digit - 10 + 'A') : (digit + '0')  //the first condition considers hex
    } while ((n /= b) > 0);  // more expressive n = n / b
    if (sign < 0) {
        s[i++] = '-';  // s[i++] loops to the next within the array string

    }
    s[i] = '\0';
    reverse(s);
}


int main() {
    char buffer[50];    // when initializing outside of a function, you'll need to include an array length

    int n = 255;
    int base = 16;
    itob(n, buffer, base);
    print("%d in base %d is %s\n", n, base, buffer);

    return 0;
}




//New Code

//if a number is stored in an array, the array is reordered from low-order digits to high-order digits

// to print properly, we can complete our computation and use the reverse_string function. or we can formulate a recursive solution.


//print n in decimal n is also a decimal.

#include <stdio.h>

void printd(int n){
    if (n < 0){
        putchar('-');
        n = -n;
    }
    if (n / 10)
        printd(n / 10);
    putchar(n % 10 + '0');  //start from the lowest multiple
}


//New Code

// A recursive version of reverse(s)

#include <stdio.h>
#include <string.h>

void reverse_recursive_helper(char s[], int start, int end){
    if (start >= end) {
        return;   //exit condition
    }

    char temp = s[start];
    s[start] = s[end];
    s[end] = temp;

    reverse_recursive_helper(s, start + 1, end - 1);
}

void reverse(char s[]){
    int len = strlen(s);

    reverse_recursive_helper(s, 0, len-1);

}


int main() {
    char str1[] = "hello";
    reverse(str1);
    printf("Reversed: %s\n", str1);
}