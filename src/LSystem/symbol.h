#ifndef SYMBOL_H
#define SYMBOL_H


class Symbol
{
public:
    Symbol(char, int);
    char m_refCharacter;
    int iteration;
    Symbol *next;
    Symbol *prev;
};

#endif // SYMBOL_H
