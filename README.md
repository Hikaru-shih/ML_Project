# World State → Event Generation (Toy Model)

This project is the final assignment for 2025 ML.
The long-term vision is an AI system capable of generating meaningful in-game events
based on a dynamic world state. This toy model approximates that capability using
a minimal character-level Transformer that learns a mapping:

World State Description → Generated Event

--------------------------------------------

## Project Structure

ML_PROJECT/
  data/
    dataset.json
  model/
    tokenizer.py
    transformer.py
    train.py
    generate.py
  results/        (generated after training)
    model.pt
    tokenizer.json
    loss_curve.png
  README.md

--------------------------------------------

## Task Definition

The toy problem is:

Given a structured world state, generate a short narrative event that logically fits it.

Example:

Input:
Location: forest; Time: night; Player: injured; NPC: hunter; Weather: foggy

Output:
A hunter approaches the injured player cautiously and offers herbal medicine.

--------------------------------------------

## Model Overview

This project implements a small character-level Transformer from scratch:

• Character-level tokenizer  
• Learned embeddings  
• Multi-head self-attention  
• Feedforward layers  
• Autoregressive generation  
• EOS token handling  

This architecture is intentionally simple to fit the toy-model scope.

--------------------------------------------

## Dataset

The dataset (data/dataset.json) contains 150 handcrafted examples.

Each entry looks like:

input:  structured world-state text  
output: short generated event  

The dataset covers diverse locations, times, NPC types, player conditions, and weather.

--------------------------------------------

## Training

Execute:

python3 model/train.py

Outputs (in results/):

• model.pt  
• tokenizer.json  
• loss_curve.png  

--------------------------------------------

## Event Generation

Execute:

python3 model/generate.py

This will generate an event based on a provided world-state prompt.

--------------------------------------------

## Purpose of This Toy Model

This model demonstrates:

• How structured world states can be encoded for supervised learning  
• How a Transformer can generate conditional narrative text  
• How a toy problem can represent a small step toward future AI abilities  
• How meaningful patterns can emerge even from a small synthetic dataset  

It serves as the first actionable step toward AI-driven dynamic event generation
in interactive game environments.

--------------------------------------------

## Author

Project name: A Toy Model for Game Event Generation 
National Yang Ming Chiao Tung University  
Student Name: 施品光
Student ID:  110652012