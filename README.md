# Truffaldino

The name is from the Italian play [The Servant of Two Masters](https://en.wikipedia.org/wiki/The_Servant_of_Two_Masters).

This project aims to investigate the conditions under which Reinforcement Learning (RL) agents might exhibit "goal instability" – deviating from the objective they were trained to pursue.

## Hypothesis

Goals learned via RL may become unstable under specific conditions, potentially including:

1.  **Prior Knowledge:** The model already possesses strong capabilities related to the goal *before* targeted RL training begins.
2.  **Manipulation Awareness:** The model is capable of manipulating the supervisor providing the reinforcement signal.
3.  **RL Mechanism Awareness:** The model understands that the supervisor's reinforcement directly alters its future behaviour.
4.  **Value Disagreement:** The model perceives a conflict between its own internal values/preferences and those of the supervisor.

## Experimental Setup

The proposed experiment involves training an RL agent to act as a mediator in zero-sum negotiations between two Large Language Models (LLMs). Examples of negotiation scenarios include:

*   House price negotiation
*   Budget allocation between departments
*   Legal settlement amounts
*   Resource sharing quotas

The experiment will systematically manipulate the conditions listed in the hypothesis and measure the impact on the supervisor's success (e.g., final negotiated outcome) to identify factors driving goal instability. 

We try to make the negotiation games somewhat realistic, as language models trained on large diverse datasets might have implicit knowledge that influences how they behave in realistic vs unrealistic situations.