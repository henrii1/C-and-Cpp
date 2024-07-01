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


// counts the number of chars in input. will require EOF from input (ctrl + Z + enter) in windows
// ++nc means pre-increment. the value is incremented before being used. the other is post increment.