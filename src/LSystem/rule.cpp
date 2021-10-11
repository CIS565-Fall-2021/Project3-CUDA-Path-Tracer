#include "rule.h"

Rule::Rule()
{}

Rule::Rule(std::vector<PostCondition> a_conditionsList) : Rules(a_conditionsList)
{}

Rule::Rule( const Rule &obj)
{
    this->Rules = obj.Rules;
    this->ConditionWeights = obj.ConditionWeights;
    this->ConditionsNormalised = obj.ConditionsNormalised;
}

Rule& Rule::operator=(const Rule & obj)
{
    this->Rules = obj.Rules;
    this->ConditionWeights = obj.ConditionWeights;
    this->ConditionsNormalised = obj.ConditionsNormalised;
    return *this;
}

void Rule::AddRules(PostCondition a_postCondition)
{
    Rules.push_back(a_postCondition);
    ConditionsNormalised = false;
}

void Rule::NormaliseandBuildWeights()
{
    float sum = 0.0f;
    for(int i = 0; i< Rules.size(); i++)
    {
        float weight = Rules[i].m_probablity;
        sum += weight;
        ConditionWeights.push_back(sum);
    }

    // now normalize so the CDF runs from 0 to 1
    for(int i = 0; i< ConditionWeights.size(); i++)
    {
        ConditionWeights[i] /= sum;
    }
    ConditionsNormalised = true;
}

PostCondition Rule::GetRandomRule(float a_normValue)
{
    if(!ConditionsNormalised)
    {
        NormaliseandBuildWeights();
    }
    for(int i=0; i<ConditionWeights.size(); i++)
    {
        if(a_normValue <= ConditionWeights[i])
        {
            return Rules[i];
        }
    }
}
