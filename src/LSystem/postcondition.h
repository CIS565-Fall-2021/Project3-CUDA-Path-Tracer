#ifndef POSTCONDITION_H
#define POSTCONDITION_H

#include<string>
class PostCondition
{
public:
    PostCondition(float, std::string);
    float m_probablity;
    std::string new_symbol;
};

#endif // POSTCONDITION_H
