#ifndef RULE_H
#define RULE_H

#include<vector>
#include "postcondition.h"

class Rule
{
public:
    Rule();
    Rule(std::vector<PostCondition>);
    Rule( const Rule &obj);
    Rule& operator=(const Rule& n);

    std::vector<PostCondition> Rules;
    std::vector<float> ConditionWeights;
    bool ConditionsNormalised;

    void AddRules(PostCondition);
    void NormaliseandBuildWeights();

    PostCondition GetRandomRule(float a_normValue);

};

#endif // RULE_H
