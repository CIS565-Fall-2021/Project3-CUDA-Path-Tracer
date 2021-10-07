#ifndef LSYSTEMS_H
#define LSYSTEMS_H

#include<unordered_map>
#include<string>
#include<stdlib.h>
#include "symbol.h"
#include<memory>
#include<iostream>
#include<vector>
#include<cmath>
#include "rule.h"
#include"turtle.h"
#include"glm/glm.hpp"
#include"glm/gtc/matrix_inverse.hpp"
#include "../sceneStructs.h"
#include "../utilities.h"
#pragma once
#include <memory>
#include<random>


// A collection of preprocessor definitions to
// save time in writing out smart pointer syntax
#define uPtr std::unique_ptr
#define mkU std::make_unique
// Rewrite std::unique_ptr<float> f = std::make_unique<float>(5.f);
// as uPtr<float> f = mkU<float>(5.f);

#define sPtr std::shared_ptr
#define mkS std::make_shared
// Rewrite std::shared_ptr<float> f = std::make_shared<float>(5.f);
// as sPtr<float> f = mkS<float>(5.f);


typedef void (Turtle::* RuleFunc)();
class LSystem
{
public:
    LSystem(Turtle&);
    uPtr<Turtle> currTurtle;

    std::string axiom;
    std::unordered_map<char, Rule> RulesList;
    std::unordered_map<char, RuleFunc> Func_Pointer;
    Symbol* rootSymbol;
    std::vector<sPtr<Symbol>> SymbolList;
    void AddRule(char, Rule);
    void AddFuncPointer(char, RuleFunc);
    void AssignAxiom(std::string);
    void LSystemParse(int iterations);
    void ClearParsedString();
    void ApplyRule(Symbol* a_prevSym, Symbol* a_currSym, Symbol* a_endSym, char a_ruleKey, int iteration);
    void GetRandomRule(char a_ruleKey);
    void CarveBuilding(std::vector<glm::vec3>& procShape, std::vector<Geom> &geoms, std::vector<Material>& materials);
    void AddDefaultFuncPointer();
    void PrintParsedSystem();
    //void digBlock(int x, int z, int depth);
};

#endif // LSYSTEMS_H
