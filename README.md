# Game Theory Final Project

## Modeling Commuter Behavior in Urban Transport Using Agent-Based Simulation and Game Theory

### Abstract
This paper presents an agent-based model to study urban commuter transportation behavior under policy interventions aimed at reducing greenhouse gas emissions. We simulate a synthetic population of 20,000 agents who choose among four commuting modes: car, bus, train, and bike/walk, based on personal attributes, travel utility, and historical preferences. Mode choice follows a multinomial logit framework, and congestion evolves endogenously as a function of car usage. Policy interventions, such as increased tolling and public fare subsidies, are introduced mid-simulation to evaluate behavioral shifts and their impact on environmental and economic externalities. Our framework enables policymakers to test policy bundles and evaluate trade-offs between emissions targets, commuter welfare, and public finance.

### Introduction
Urban transportation systems are complex sociotechnical networks in which individual commuter decisions aggregate to produce large-scale outcomes such as congestion, emissions, and infrastructure costs. As cities face increasing pressure to meet climate goals, reduce congestion, and maintain financial sustainability, understanding how commuters respond to policy changes is vital. Traditional forecasting tools often assume static behavior or overlook the feedback loops inherent in transportation systems.

In this work, we develop an agent-based simulation model to investigate how commuters adapt their mode choices in response to cost, time, and habitual preferences, and how these choices are influenced by policies targeting externalities. The model allows us to explore questions such as:
- What policy combinations are most effective in shifting commuters away from private vehicles?
- How do income and value-of-time heterogeneity affect policy outcomes?
- Can a system reach emissions targets without compromising social welfare?

Agent-based models (ABMs) have been widely used to study transportation behavior due to their ability to represent individual heterogeneity and system-level feedback. Previous work has applied ABMs to traffic flow [Balmer et al., 2004], car ownership modeling [Zhang et al., 2005], and policy evaluation [Horni et al., 2016]. Multinomial and nested logit models remain foundational in travel demand modeling [Train, 2009], but are often incorporated into static or aggregate-level frameworks.

Recent advances have integrated behavioral economics and game theory into ABMs to study modal shift [Kumar & Bierlaire, 2012], but few models combine real-world income distributions, congestion feedback, and policy interventions within a unified simulation. Our approach builds on these foundations by incorporating empirical data, dynamic feedback, and economic accounting to support decision-making for urban mobility policy.

### Methods
We implement an agent-based model (ABM) of commuter mode choice in an urban environment using the Mesa framework in Python. The model simulates a synthetic population of 20,000 agents representing urban commuters with heterogeneous socioeconomic attributes. Agents repeatedly make transportation decisions among four modes—car, bus, train, and bike/walk—based on travel utility, cost, and historical preferences. Policy interventions are introduced mid-simulation to assess behavioral responses and externalities such as greenhouse gas (GHG) emissions and total system cost.

#### Agent Formulation
Each agent is assigned an income drawn from a Pareto distribution parameterized to match a specified median and mean income for the region (Philadelphia). From income, the agent’s car ownership probability is determined using a logistic function.

Agents are categorized into socioeconomic groups—lower, middle, and upper class—based on thresholds relative to the median income. Commute distances are sampled from log-normal distributions with group-specific parameters. Price sensitivity is inversely proportional to income, while value of time is approximated as half the agent’s hourly wage (income divided by 2080 hours/year).

#### Utility and Mode Choice
Mode choice follows a multinomial logit model, with utilities calculated as:

- **ASC**: Alternative-specific constant, calibrated using income-stratified Philadelphia mode share data
- **Habit**: A car-specific stickiness bonus increasing with consecutive car use
- **Beta_p**: Price sensitivity (income-scaled)
- **Beta_t**: Value of time
- **Change in C, Change in T**: Deviations from baseline Cost and Time (calculated at model initialization with real-world data)

Commuting time is computed as mode-specific minutes per mile multiplied by trip distance. Car and bus travel times are further scaled by a congestion factor.

Agent decisions are updated monthly. Mode probabilities are computed via softmax over utilities, and agents stochastically sample their choice. Car usage contributes to a cumulative habit streak, increasing future car utility.

#### Congestion and Feedback
Congestion is endogenously calculated based on the share of agents driving.

Travel times for cars and buses are multiplied by this congestion factor, introducing a feedback loop that penalizes excessive car use.

#### Policy Interventions
Two primary policy levers are modeled:

- **Car Toll Adjustment**: An additional toll applied to driving beyond baseline real-world estimates
- **Public Transit Fare Discount**: A proportional discount applied to bus and train fares

Policy combinations are introduced after 12 simulated months. The model then runs for an additional 12 months to capture adaptation dynamics.

#### Externalities and Costs
Total emissions come from both cars and trucks, under both free-flow and congested conditions where H terms are total hours traveled and e terms are emissions rates per hour.

Congestion costs combine time and fuel costs due to slower travel speeds where c is the fuel cost per hour and v is the value of time lost in traffic.

Public and private transport systems are evaluated together in a net fiscal balance where fare revenue and transit cost are based on ridership and per-rider values, while tolls and maintenance are based on car usage and fixed cost rates.

We track all commuting time across the population where T is the total commute time per agent, including delays from congestion.

### Results/Future Work
At this stage of the project, the simulation is able to reasonably replicate the observed transportation mode breakdown for Philadelphia without explicitly hardcoding most behavioral outcomes. The only hardcoded parameters are income distributions, commute distances (based on empirical distributions), and the current mode share baselines used to initialize utility constants. Mode choices and emergent outcomes like congestion and emissions evolve organically from the agents’ individual attributes and decision logic.

The early results are promising in that they show realistic aggregate patterns even without fine-tuning. For example, lower-income agents tend to prefer buses and biking/walking due to higher price sensitivity, while higher-income agents are more likely to drive or take the train—aligning with real-world expectations.

One limitation so far is that the model operates at a fixed population size. A logical next step would be to introduce gradual population growth to reflect urban trends and assess long-term sustainability. Additionally, more extensive simulations of different policy scenarios are needed to draw strong conclusions. However, this is currently limited by computational load, especially when running long-term simulations with large agent populations.

Overall, the model lays a solid foundation for policy testing, and with further parameter sweeps and optimization, it could be a useful tool for evaluating trade-offs between emissions targets, commuter welfare, and financial sustainability.

### References
- Balmer, M., Axhausen, K. W., & Nagel, K. (2004). An agent-based demand-modeling framework for large-scale microsimulations. *Transportation Research Record: Journal of the Transportation Research Board, 1898*(1), 125–134. https://doi.org/10.3141/1898-15
- Horni, A., Nagel, K., & Axhausen, K. W. (Eds.). (2016). *The multi-agent transport simulation MATSim*. Ubiquity Press. https://doi.org/10.5334/baw
- Kumar, N., & Bierlaire, M. (2012). Simulation based framework to model dynamic decisions of travelers in a multimodal transportation network. In *Procedia - Social and Behavioral Sciences, 54*, 975–984. https://doi.org/10.1016/j.sbspro.2012.09.803
- Train, K. (2009). *Discrete choice methods with simulation* (2nd ed.). Cambridge University Press. https://doi.org/10.1017/CBO9780511805271
- Zhang, W., Levinson, D., & Zhu, S. (2005). Agent-based model of household vehicle decision-making. *Transportation Research Record: Journal of the Transportation Research Board, 1921*(1), 29–36. https://doi.org/10.1177/0361198105192100104
- Pew Charitable Trusts. (2019). The state of commuters in Philadelphia, 2019. https://www.pewtrusts.org/-/media/assets/2019/04/philadelphia_state_of_commuters.pdf
- Southeastern Pennsylvania Transportation Authority (SEPTA). (n.d.). Open Data Portal. https://www.septa.org/open-data/
- U.S. Census Bureau. (2020a). QuickFacts: Philadelphia city, Pennsylvania. https://www.census.gov/quickfacts/philadelphiacitypennsylvania
