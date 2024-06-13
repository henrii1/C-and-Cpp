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
        