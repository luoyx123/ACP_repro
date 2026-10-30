JOURNALSUBMISSION 1
AME-2: Agile and Generalized Legged Locomotion via
Attention-Based Neural Map Encoding
Chong Zhang123∗, Victor Klemm1, Fan Yang1, Marco Hutter1
Website: sites.google.com/leggedrobotics.com/ame-2
Abstract—Achieving agile and generalized legged locomotion errors in estimation and mapping, especially under visual oc-
across terrains requires tight integration of perception and clusionsandviolationsofmodelingassumptions[4].Elevation
control,especiallyunderocclusionsandsparsefootholds.Existing
maps are often updated at lower rates than the control loop
methodshavedemonstratedagilityonparkourcoursesbutoften
and heuristically filtered [2], [4], [11], which often requires
rely on end-to-end sensorimotor models with limited generaliza-
tionandinterpretability.Bycontrast,methodstargetinggeneral- per-terrain tuning to ensure safety and makes it difficult to
izedlocomotiontypicallyexhibitlimitedagilityandstrugglewith obtainageneralsolutionfordiverseterrainsthatdemandagile
visual occlusions. We introduce AME-2, a unified reinforcement motions.Moreover,theoptimizationandplanningcomponents
learning (RL) framework for agile and generalized locomotion
in these pipelines tend to be computationally heavy, which
that incorporates a novel attention-based map encoder in the
can limit agility on real robots. These limitations motivate
control policy. This encoder extracts local and global mapping
features and uses attention mechanisms to focus on salient learning-based approaches such as RL that can better exploit
regions, producing an interpretable and generalized embedding raw sensor data, operate under uncertainty, and produce more
for RL-based control. We further propose a learning-based dynamic behaviors.
mapping pipeline that provides fast, uncertainty-aware terrain
Learning-based methods, in particular RL, have recently
representations robust to noise and occlusions, serving as policy
shown strong potential for legged locomotion [12]. One line
inputs.Itusesneuralnetworkstoconvertdepthobservationsinto
localelevationswithuncertainties,andfusesthemwithodometry. of work combines RL with classical state estimation and
The pipeline also integrates with parallel simulation so that we mappingpipelines,usingelevationmapsorrelatedrepresenta-
can train controllers with online mapping, aiding sim-to-real tions as policy inputs [13]–[26]. These methods can improve
transfer.WevalidateAME-2withtheproposedmappingpipeline
robustness and generalization compared to purely model-
on a quadruped and a biped robot, and the resulting controllers
based controllers, but they also inherit the computational cost
demonstrate strong agility and generalization to unseen terrains
in simulation and in real-world experiments. and failure modes of the underlying estimation and mapping
systems [27], [28]. In simulation, noise is typically injected
Index Terms—Legged Robots, Humanoid and Bipedal Loco-
into both the state estimates and the mapping observations. A
motion, Motion Control, Reinforcement Learning.
commonstrategyistoassumeafullyobservedegocentricmap
during training and then tune the real-world mapping stack to
I. INTRODUCTION
approximate this assumption [13], [15], [21], which simplifies
AGILE and generalized locomotion is essential for legged training but can make performance sensitive when occlusions
robots to operate reliably in diverse real-world environ- occur or when agile whole-body contacts violate estimator
ments. It requires tightly coupled real-time perception and assumptions. Another strategy is to heavily randomize state-
control, robustness to sensor noise and occlusions, agility estimation and mapping noise so that the controller becomes
through whole-body control, and precise behavior on terrains robust and conservative under mapping uncertainties in dense
with sparse footholds. Designing such perceptive locomotion terrains[14],yetsuchpoliciestendtostruggleonterrainsthat
systems that are simultaneously agile and able to generalize require very high agility or precise foothold placement.
across diverse terrains remains a critical challenge. More recently, neural network–based mapping has been
Classical pipelines address perceptive locomotion by com- explored, where networks are trained in simulation to re-
biningmodel-basedcontrolwithexplicitmappingandstatees- construct maps that can be used as policy inputs [29]–[34].
timation[1]–[8].Typically,theymaintainastateestimator[9], While this yields efficient mapping suitable for real-time
build elevation maps from sensor data [10], [11], and use control, the learned mappings are usually fitted to particular
model-based planning and control methods to command the terrain distributions and sensor configurations, offer limited
robot based on this mapping information. These approaches generalization to unseen terrains, and do not explicitly model
have proven effective for deliberate walking and carefully uncertainties such as occlusions.
planned maneuvers on moderately structured terrain, where Another line of work directly maps raw exteroceptive per-
accuratestateestimationanddense,slowlychangingmapscan ception and proprioception to actions [35]–[47]. These sen-
be maintained. However, their reliance on precise models and sorimotor policies have demonstrated highly agile behaviors
often deterministic optimal control makes them sensitive to on challenging courses [36], [38], [42], but typically exhibit
limited generalization beyond the training environments and
1RoboticSystemsLab,ETHZurich,Switzerland
offer little interpretability, since terrain reasoning is implicitly
2Secure,Reliable,andIntelligentSystemsLab,ETHZurich,Switzerland
encoded in the policy. More recent sensorimotor general-
3ETHAICenter,Switzerland
∗Corresponding:chong.zhang@ai.ethz.ch ist models [48] distill a single policy from multiple expert
6202
naJ
31
]OR.sc[
1v58480.1062:viXra

JOURNALSUBMISSION 2
Fig.1. Ourmethodenablesagileandgeneralizedleggedlocomotionacrossdiverseterrainswithonboardsensingandcomputation.
controllers to achieve strong performance, yet such methods mappingstackswhilemaintaininggeneralization.Thepipeline
still show limited generalization to unseen terrains and of- projects depth images into local grids and uses a lightweight
ten require finetuning on new terrains, as evaluated in our neural network trained via Bayesian learning [51] to predict
benchmarks(Sec.VII).Overall,existingRL-basedapproaches localelevationswithper-celluncertaintyestimates.Theselocal
tend to trade off agility, generalization, mapping efficiency, maps are fused into a consistent global frame with odometry,
and interpretability rather than addressing these requirements providing a fast and uncertainty-aware representation that
within a unified framework. accounts for occlusions and sensor noise. We then query
In this work, we present AME-2, a unified RL framework egocentric elevations and associated uncertainties as the map
forperceptiveleggedlocomotionwithanAttention-basedMap inputs for the controller. This design is inspired by [52],
Encoding architecture trained jointly with the controller. The which fuses per-frame map predictions instead of training
AME-2 encoder, built upon the design in [15], first extracts temporal networks for off-road navigation, thereby reducing
local features(pixel-wise representationof terrain details) and data requirements and mitigating overfitting. Furthermore, we
global features (capturing global terrain context) from the synthesize random terrain data, including many terrains that
elevation map, then uses the global features together with are not feasible for the robots, to broaden the training distri-
proprioceptiontoassignattentionweightstothelocalfeatures. butionforthelocalmappredictorandimprovegeneralization.
The resulting weighted local features are concatenated with Thesamemappingpipelinerunsbothinparallelsimulation
the global features and proprioception to form a terrain-aware andontherealrobots,enablingonlinemappingduringtraining
representation for policy learning. This design allows the ratherthanrelyingonidealizedorhand-tunedmaps.Fortrain-
policy to downweigh local regions that are less relevant for ing efficiency, we adopt a teacher–student scheme [48], [53]:
the current task, improving generalization to new terrains. a teacher policy is first trained with ground-truth elevation
Because the attention is conditioned on global terrain context, maps, and a student policy with the same AME-2 architecture
thepolicycanalsolearndistinctattentionandmotionpatterns is then trained using the proposed mapping under teacher
across different terrains. To enable learning agile controllers supervision alongside RL. The resulting student policy is
with this representation, we adopt a goal-reaching locomotion directlydeployable,operateswiththelightweight,uncertainty-
formulation from prior works [20], [26], [32], [40], [49], [50] aware mapping in the loop, and retains the generalization and
and adapt it so that the same reward functions and training agility of the teacher.
settings can be used for different robots. Together, the AME- We evaluate AME-2 on an ANYmal-D quadruped [54] and
2 encoder and unified training formulation enable training a a LimX TRON1 biped [55] both in simulation and on real
single generalist policy with terrain-aware skills, which in our hardware.AsshowninFig.1,theresultingcontrollersexhibit
experiments exhibits strong generalization to unseen terrains strong agility and robust generalization to a wide range of
while maintaining high agility. terrains.Theproposedmappingpipelineproduceshigh-quality
To enable real-world deployment of agile and generalized elevation maps in real time, supporting these agile motions
locomotion controllers, we develop a learning-based elevation with onboard perception and computation.
mapping pipeline to remove the dependency on classical The main contributions of this work are summarized as

JOURNALSUBMISSION 3
follows: environments. In contrast, our work combines a fully RL-
1) We propose AME-2, a unified RL framework with based controller with a learned mapping module designed for
an attention-based map encoder to achieve agile and generalization. As a result, our policies achieve agility com-
generalized legged locomotion. parable to previous state of the art while enabling generalized
2) Wedevelopalightweight,uncertainty-awareelevation locomotion across diverse terrains.
mapping pipeline that generalizes to diverse terrains,
explicitlymodelsocclusionsandnoise,andhelpsbridge B. Perceptive Locomotion with Raw Sensor Data
the sim-to-real gap.
Perceptivelocomotionwithrawsensordataseekstobypass
3) Weuseateacher-studentschemethatyieldsdeployable
thelimitationsofexplicitmapping.Insteadofconstructingand
controllers using the learned mapping while retaining
maintaining a map, some works directly train policies from
agility, generalization, and interpretability.
cameraimages[35]–[39],[41]–[47]orlidarpointclouds[40].
4) We demonstrate a state-of-the-art combination of
Early approaches in this direction still separate perception
agilityandgeneralizationwithaquadrupedandabiped
and control: a visual module outputs footholds or trajectories,
robotacrossdiversechallengingterrainsunderthesame
which are then tracked by model-based controllers [57]. Be-
training setup.
tween explicit mapping and fully end-to-end policies, other
works learn intermediate representations from raw sensors
II. RELATEDWORKS (such as ray distances [49] and neural volumetric mem-
A. Perceptive Locomotion with Explicit Mapping ory [58]) that remain cheap to compute while enabling fast
control.
Early explorations of perceptive locomotion use model-
Many recent methods adopt a monolithic design, where a
based methods with either prebuilt maps or classical online
single neural network maps raw sensor streams and proprio-
mapping. Most of them first perform foothold or trajectory
ception directly to joint commands. Within this family, some
planningandthenapplyamodel-basedtrackingcontroller[1]–
distill deployable student policies from privileged teacher
[3],[5]–[8],whilesomedirectlyoptimizeareactivecontroller
policies—oftenusingmapsasprivilegedinputs—toachieve
over feasible footholds [4]. Although these methods offer
efficient and stable learning [35], [36], [38], [42], [43], [46],
strongguaranteesandcangeneralizewellundertheirmodeling
[48], while others learn from scratch with the help of repre-
assumptions,theirreal-worldperformanceislimitedbymodel
sentation learning techniques such as world models [44] or
mismatch, state-estimation uncertainties, mapping errors, and
privileged-information reconstruction [41], [45]. The current
computational burden.
state of the art in agile generalist locomotion [48] trains
To address these limitations, RL emerges as an alternative
multiple mapping-based teacher controllers and distills them
for perceptive locomotion [12]. One line of work keeps a
intoasinglegeneraliststudentpolicythatoperatesdirectlyon
strongmodel-basedcomponentandusesRLmainlyasanadd-
raw depth images. This student policy combines the skills of
on[21],[22],[56].DTC[21]augmentsamodel-basedplanner
the teachers and can be finetuned efficiently on new terrains.
with an RL tracking controller to improve robustness while
However, its zero-shot generalization to previously unseen
maintaining generalization, but the planner remains sensitive
environments is limited.
to mapping uncertainties and constrains agility through its
In contrast, our work also trains neural networks from raw
modeling assumptions and computational cost. RLOC [22]
sensor data, but uses them exclusively within the mapping
instead uses a model-based controller to track an RL planner,
processratherthaninamonolithicperception-to-actionpolicy.
yettheoverallperformanceiscappedbythelimitationsofthe
Because our learned mapping is lightweight and well suited
controller.
to massive parallelization in simulation, we can run the full
Otherapproachesremovethemodel-basedmodulesentirely
mappingloopduringtrainingandthendeployexactlythesame
while still relying on explicit maps. When given prebuilt
mapping at test time, preserving both speed and consistency.
maps, RL controllers can exhibit highly agile motions [26].
Without training a monolithic policy that directly maps raw
With classical online mapping, they outperform model-based
sensor data to joint actions, we still obtain an agile controller
counterpartsintermsofagilityandrobustness[13]–[19],[23],
while also achieving strong generalization across diverse ter-
[24], but their agility remains limited by the update rate and
rains.
quality of the mapping. In addition, many such controllers
suffer from generalization issues, as they tend to overfit to the
C. Learning-Based Mapping
training terrains [21]. A recent work [15] shows that RL can
achieve generalized locomotion across diverse terrains, but its Learning-based mapping aims to retain the structure and
agilityisstillconstrainedbytheonlineclassicalmappingstack interpretabilityofexplicitmapswhileovercomingthecompu-
and the learning framework. tational and modeling limitations of classical mapping. Clas-
Morerecently,RL-basedmethodshavebeguntouselearned sical mapping pipelines combine geometric projection, hand-
maps, where neural networks reconstruct egocentric terrains tunedfiltering,andprobabilisticfusiontoproducehigh-quality
for the controller in real time [29]–[34]. These approaches maps, and can handle a wide range of complex terrains [10],
demonstrate strong agility and successful real-world deploy- [11],[23],[24].However,thesemethodsareoftentooslowfor
ment, but the learned maps often fit the training distribu- agilelocomotionandrequireheuristic,terrain-specificfiltering
tion and show limited generalization in unseen, unstructured to cope with occlusions and sensor noise.

| JOURNALSUBMISSION |     |     |     |     |     |     |     |     |     |     |     |     |     |     | 4   |
| ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Simulation
 RL Optimization
Proprioception
|     |     |     |     | Privileged Teacher Policy |     |     |     | Joint Actions |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ------------------------- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
Ground-Truth Mapping
Supervision
 RL Optimization
Proprioception
|     |     |     |     | Deployable Student Policy |     |     |     | Joint Actions |     |     |     |     |     |     |     |
| --- | --- | --- | --- | ------------------------- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- | --- | --- |
Online Neural Mapping
Depth Cloud & Odometry
Fig.2. Anoverviewofoursystem.WeuseRLtotrainateacherpolicywithground-truthmappinginsimulation,andastudentpolicyundertheteacher’s
supervisionwithourproposedneuralmapping.Thepoliciesoutputjoint-levelactionsthatactuatetherobotstoreachpositionandheadinggoals,asillustrated
bytheyellowwireframes.
To enable faster control, several works directly train ter- PPO [60] in parallel simulation [61]. By maximizing goal-
rain reconstruction modules in or after the policy training reaching rewards (detailed in Sec. IV), the policies learn agile
loop[29]–[34].Theselearnedmappingscaninferlocalgeome- locomotion skills that enable them to traverse diverse terrains.
tryquicklyenoughtosupportagilecontrollers,butareusually We first train a privileged teacher policy and then transfer the
trained on a narrow distribution of environments, showing skills to a deployable student policy, which is also detailed in
| limited generalization |     |     | beyond | the training | terrains. |     |     | Sec. IV. |     |     |     |     |     |     |     |
| ---------------------- | --- | --- | ------ | ------------ | --------- | --- | --- | -------- | --- | --- | --- | --- | --- | --- | --- |
Recentoff-roadnavigationapproachestakeadifferentroute
| to obtain | uncertainty-aware |             |     | neural         | mapping | with | improved  |                 |        |        |         |            |         |       |       |
| --------- | ----------------- | ----------- | --- | -------------- | ------- | ---- | --------- | --------------- | ------ | ------ | ------- | ---------- | ------- | ----- | ----- |
|           |                   |             |     |                |         |      |           | B. Controller   | Inputs | and    | Outputs |            |         |       |       |
| accuracy  | and data          | efficiency. |     | In particular, |         | [52] | leverages |                 |        |        |         |            |         |       |       |
|           |                   |             |     |                |         |      |           | Our controllers |        | run at | 50 Hz.  | The policy | actions | a are | joint |
neuralprocesses[59]tomodelterrainelevation:anetworkpre-
|                                                        |       |                 |      |            |          |       |            | PD targets       | [62] | tracked      | at 400    | Hz on real | hardware.   |          |     |
| ------------------------------------------------------ | ----- | --------------- | ---- | ---------- | -------- | ----- | ---------- | ---------------- | ---- | ------------ | --------- | ---------- | ----------- | -------- | --- |
| dicts per-frame                                        |       | local elevation |      | maps       | together | with  | associated |                  |      |              |           |            |             |          |     |
|                                                        |       |                 |      |            |          |       |            | Proprioception   |      | observations |           | include    | base linear | velocity | v b |
| uncertainties,                                         | which | are             | then | fused over | time     | using | odometry.  |                  |      |              |           |            |             |          |     |
|                                                        |       |                 |      |            |          |       |            | (only accessible |      | to the       | teacher), | base       | angular     | velocity | ω , |
| Bypredictinglocalmapswithuncertaintyinsteadoflearninga |       |                 |      |            |          |       |            |                  |      |              |           |            |             |          | b   |
|                                                        |       |                 |      |            |          |       |            |                  |      | g            |           |            | q,          |          | q˙, |
fully end-to-end sensor-to-map model, this approach reduces projected gravity b , joint positions joint velocities
|          |              |     |              |            |     |     |          | previous actions, |     | and goal | commands   | c.   |       |        |      |
| -------- | ------------ | --- | ------------ | ---------- | --- | --- | -------- | ----------------- | --- | -------- | ---------- | ---- | ----- | ------ | ---- |
| the data | requirements |     | for accurate | estimation |     | and | improves |                   |     |          |            |      |       |        |      |
|          |              |     |              |            |     |     |          | Ground-truth      |     | mapping  | represents | each | point | in the | ego- |
generalizationincomplexoff-roadenvironments.However,the
|               |       |     |            |     |      |              |       | centric elevation |     | grid | by its | 3D coordinates |     | (x,y,z). | Our |
| ------------- | ----- | --- | ---------- | --- | ---- | ------------ | ----- | ----------------- | --- | ---- | ------ | -------------- | --- | -------- | --- |
| computational | costs | of  | the models | in  | [52] | are too high | (each |                   |     |      |        |                |     |          |     |
neuralmappingaugmentsthiswithanuncertaintychanneland
| environment | takes | >   | 13 GB | GPU | memory) | and | cannot be |     |     |     |     |     |     |     |     |
| ----------- | ----- | --- | ----- | --- | ------- | --- | --------- | --- | --- | --- | --- | --- | --- | --- | --- |
producesa4Drepresentation(x,y,z,u)foreachpoint,where
| deployed       | with thousands |        | of       | parallel simulation |                  | environments. |      |                      |     |         |     |     |     |     |     |
| -------------- | -------------- | ------ | -------- | ------------------- | ---------------- | ------------- | ---- | -------------------- | --- | ------- | --- | --- | --- | --- | --- |
|                |                |        |          |                     |                  |               |      | u is the uncertainty |     | metric. |     |     |     |     |     |
| Inspired       | by these       | ideas, |          | we design           | a learning-based |               | ele- |                      |     |         |     |     |     |     |     |
| vation mapping |                | module | tailored | for                 | agile legged     | locomotion.   |      |                      |     |         |     |     |     |     |     |
IV. LOCOMOTIONCONTROL
Weprojectdepth-camerapointcloudsintolocalelevationgrids
andtrainlightweight,robustnetworkstopredictlocalelevation A. Policy Architecture and AME-2 Encoder
anditsuncertaintyfromtheseinputs,whicharethenfusedover
|            |           |      |        |            |     |        |           | The general | policy | architecture |     | is illustrated |     | in Fig. | 3, with |
| ---------- | --------- | ---- | ------ | ---------- | --- | ------ | --------- | ----------- | ------ | ------------ | --- | -------------- | --- | ------- | ------- |
| time using | odometry. | This | design | suppresses |     | sensor | noise and |             |        |              |     |                |     |         |         |
theAME-2encoderservingasthecorefeatureextractorforthe
| naturally  | models  | occlusions     |     | as regions | of  | high uncertainty. |         |         |        |         |                  |     |         |     |       |
| ---------- | ------- | -------------- | --- | ---------- | --- | ----------------- | ------- | ------- | ------ | ------- | ---------------- | --- | ------- | --- | ----- |
|            |         |                |     |            |     |                   |         | mapping | input. | We have | a proprioception |     | encoder | to  | embed |
| We further | improve | generalization |     | through    |     | extensive         | random- |         |        |         |                  |     |         |     |       |
proprioceptiveobservations,andtheAME-2encodertoembed
| ization in | a custom | data | pipeline. | The | resulting |     | module | is                   |     |       |     |                    |     |            |     |
| ---------- | -------- | ---- | --------- | --- | --------- | --- | ------ | -------------------- | --- | ----- | --- | ------------------ | --- | ---------- | --- |
|            |          |      |           |     |           |     |        | the map observations |     | based | on  | the proprioception |     | embedding. |     |
lightweightenoughtoruninmassivelyparallelsimulationand
|             |     |         |        |       |         |                |     | We feed   | both    | the proprioception |     | embedding  |       | and the | map    |
| ----------- | --- | ------- | ------ | ----- | ------- | -------------- | --- | --------- | ------- | ------------------ | --- | ---------- | ----- | ------- | ------ |
| fast enough | to  | support | highly | agile | motions | in deployment, |     |           |         |                    |     |            |       |         |        |
|             |     |         |        |       |         |                |     | embedding | through | a multilayer       |     | perceptron | (MLP) | to      | output |
whileprovidinganuncertainty-awareterrainrepresentationfor
|     |     |     |     |     |     |     |     | the actions. |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- | --- |
our controllers. The AME-2 encoder first extracts local map features with
|     |     |     |     |     |     |     |     | a convolutional |     | neural | network | (CNN) | and | computes | a po- |
| --- | --- | --- | --- | --- | --- | --- | --- | --------------- | --- | ------ | ------- | ----- | --- | -------- | ----- |
III. SYSTEMOVERVIEW sitional embedding for each point with an MLP. These are
|     |     |     |     |     |     |     |     | then fused | by another | MLP | to  | obtain pointwise |     | local features. |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | ---------- | --- | --- | ---------------- | --- | --------------- | --- |
A. Formulation
|     |     |     |     |     |     |     |     | Next, an | additional | MLP | followed | by  | max pooling | over | the |
| --- | --- | --- | --- | --- | --- | --- | --- | -------- | ---------- | --- | -------- | --- | ----------- | ---- | --- |
As illustrated in Fig. 2, we use RL to train terrain-aware pointwise features produces global features that capture the
locomotion policies that reach position and heading goals. overall terrain context. We combine these globalfeatures with
We formulate the problem as a partially observable Markov a proprioceptive embedding through an MLP to obtain a
decision process (POMDP) and optimize the policies using query vector, which is used in a multi-head attention (MHA)

JOURNALSUBMISSION 5
Teacher
| Proprioception |     |     | Proprioception Encoder |     |     |     |     |     |     |     |     |     |
| -------------- | --- | --- | ---------------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Base Lin. Vel.
Base Ang. Vel.
Proj. Gravity
|     |     |     | Proprioception Embedding |     |     |     |     | MLP Decoder | Actions |     |     |     |
| --- | --- | --- | ------------------------ | --- | --- | --- | --- | ----------- | ------- | --- | --- | --- |
Joint Pos.
Joint Vel.
Prev. Actions
|     | Mapping |     | AME-2 Encoder |     |     |     | Map Embedding |     |     |     |     |     |
| --- | ------- | --- | ------------- | --- | --- | --- | ------------- | --- | --- | --- | --- | --- |
Commands
MLP
|     |     | Pointwise Local Features |     |     |     |     |     | Global Features |     |     |     |     |
| --- | --- | ------------------------ | --- | --- | --- | --- | --- | --------------- | --- | --- | --- | --- |
MLP & MaxPool
Proprioception Embedding
Weighted
|     |                      |     | MLP             |     |     | Multi-Head Attention |     | Local Features |     | Student |     |     |
| --- | -------------------- | --- | --------------- | --- | --- | -------------------- | --- | -------------- | --- | ------- | --- | --- |
|     |                      |     |                 |     |     | V                    | K Q |                |     |         |     |     |
|     | Positional Embedding |     | Local Embedding |     |     |                      |     |                |     |         |     |     |
LSIO
|     | MLP |     |     | CNN |     |     |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
MLP
MLP
|     |     |         |     | (remove  |  channels) |     |                          |     |     |     |                          |     |
| --- | --- | ------- | --- | -------- | ---------- | --- | ------------------------ | --- | --- | --- | ------------------------ | --- |
|     |     | Mapping |     |          |            |     | Proprioception Embedding |     |     |     |                          |     |
|     |     |         |     |          |            |     |                          |     |     |     | Proprioception Embedding |     |
Fig. 3. An illustration of our AME-2 policy architecture. Left top: The general abstract of our policies. Proprioceptive observations are encoded into a
proprioceptionembedding,whichisusedtogetherwiththemappingastheinputsoftheAME-2encodertoproducethemapembedding.Theproprioception
embedding and the map embedding are then concatenated and fed into an MLP to generate actions. Left bottom: The AME-2 encoder design. We extract
bothlocalfeaturesandglobalfeaturesfromthemap,andthenusetheglobalfeaturesandtheproprioceptionembeddingtoproducetheattention-weighted
localfeatures.LandW arethelengthandwidthofthemap,dmap isthedimensionofmaprepresentation(3fortheteacher,4forthestudent),dPE isthe
dimensionofproprioceptionembedding.Right:Theproprioceptionencoderdesigns.Wehavedifferentdesignsforteacherandstudentpoliciestofacilitate
sim-to-realtransfer,whilebothdesignscanfitintoouroverallarchitecture.
module [63] with the pointwise local features serving as keys the student’s proprioception embedding.
| and            | values. This | yields    | a weighted |          | local feature    |          | embedding   |               |            |              |          |               |
| -------------- | ------------ | --------- | ---------- | -------- | ---------------- | -------- | ----------- | ------------- | ---------- | ------------ | -------- | ------------- |
| that focuses   | on           | important | terrain    | regions  | based            | on       | the current |               |            |              |          |               |
| proprioceptive | state        | and       | global     | context. | Finally,         |          | the global  |               |            |              |          |               |
|                |              |           |            |          |                  |          |             | B. Asymmetric | Actor      | Critic       |          |               |
| features       | and weighted | local     | features   |          | are concatenated |          | to form     |               |            |              |          |               |
| the map        | embedding    | fed       | into the   | policy’s | action           | decoder. |             |               |            |              |          |               |
|                |              |           |            |          |                  |          |             | We adopt      | asymmetric | actor–critic | training | [67]. For the |
This design builds on the attention-based map encoder in critic,wedonotusethesameattention-baseddesignasinthe
previous work [15], which we refer to as AME-1 for clarity. actor, because the critic does not need to generalize beyond
Unlike the AME-1 encoder, our AME-2 encoder additionally the training terrains and optimizing an MHA module with
computesglobalfeaturesandusesthemtoweighthelocalfea-
|     |     |     |     |     |     |     |     | (L×W) local | feature | inputs is | costly. Instead, | we use the |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------- | ------- | --------- | ---------------- | ---------- |
tures.Thisenablesmoregeneralizedlocomotionovercomplex mixture-of-experts(MoE)designfrom[68],whichispowerful
terrains, where motion patterns should vary with the terrain. for function fitting yet much more computationally efficient
WedemonstratetheresultingperformancegapbetweenAME- to optimize. Note that, although we can train a generalized
1 and AME-2 encoders in Sec. VI and Sec. VII. teacher with our policy architecture and an MoE critic, an
The choice of proprioception encoder in our architecture MoE actor does not yield a generalized teacher, as shown in
| is flexible, | as long | as it | produces | an  | effective | proprioception |     | Sec. VII. |     |     |     |     |
| ------------ | ------- | ----- | -------- | --- | --------- | -------------- | --- | --------- | --- | --- | --- | --- |
embedding. Hence, we use different designs for the teacher Ontopofthenoiselessproprioceptiveobservationsusedby
and the student. For the teacher, since ground-truth propri- the teacher actor and the ground-truth map, we additionally
oceptive observations are available, we use a plain MLP provide the contact state of each link to the critic. We do not
over them to obtain the proprioception embedding. For the provide such contact information to the teacher, since we can
student, because of various environmental uncertainties and already train a robust, generalized teacher without it, and a
followinglessonsfrompreviousworks[64]–[66],westackthe larger information gap between teacher and student can hurt
proprioceptive observations (except base linear velocities and student training [69]. We also apply left-right symmetry aug-
commands) from the past 20 steps and use Long-Short I/O mentation from [32] to the critic to improve sample efficiency
(LSIO) [66] to obtain a temporal embedding of both robot and motion style, but not to the actor to avoid additional
states and environment dynamics. This temporal embedding computational cost of actor optimization. The critic design is
andthecommandsarethenjointlyfedintoanMLPtoproduce shared for the teacher and the student.

JOURNALSUBMISSION 6
TABLEI
REWARDSFORCONTROLLERTRAINING.
RewardTerm Expression Weight(×dτ∗) Notes
TaskRewards
PositionTracking Eq.(1) 100
HeadingTracking Eq.(3) 50
MovingtoGoal Eq.(4) 5
StandingatGoal Eq.(5) 5
RegularizationandPenalties
EarlyTermination 1ifearlyterminationtriggered −10/dτ
UndesiredEvents 1foreachundesiredevent −1
BaseRollRate [ω b ]2 x −0.1
JointRegularization ∥q˙∥2+0.01∥τ∥2+0.001∥q¨∥2 −0.001 τ denotesthejointtorques
ActionSmoothness ∥at−at−1∥2 −0.01
LinkContactForces ∥max(Fcon−G,0)∥2 −0.00001 Fcon denotesthecontactforcesforeachlinkand
Gistherobotweight
LinkAcceleration (cid:80) l ∥v˙ l ∥ −0.001 v l isthevelocityoflinkl;summingoveralllinks
SimulationFidelity summingoveralljoints
JointPositionLimits (cid:80)
j
max(0,qj−0.95q
j
max,0.95q
j
min−qj) −1000 q
j
max andq
j
min arepositionlimitsforjointj
JointVelocityLimits (cid:80)
j
max(0,|q˙j|−0.9q˙
j
max) −1 q˙
j
max isthevelocitylimitforjointj
JointTorqueLimits (cid:80)
j
max(0,|τj|−0.8τ
j
max) −1 τ
j
max isthetorquelimitforjointj
∗dτ =0.02secisthepolicyinterval.
C. Teacher-Student RL Intuitively, these two tracking rewards encourage the robot
to be at the goal position with the desired heading at the
We use Teacher–Student RL [14], [48], [53], [70] to facil-
end of the episode, without constraining how it reaches the
itate sim-to-real transfer. Although directly training a student
goal, thereby allowing complex locomotion skills to emerge
policy with our mapping pipeline is also possible, simulation
for traversing terrains.
canrunatonlyabouthalfthespeedcomparedtousingground-
To further facilitate exploration, we introduce a moving-to-
truth mapping and requires significantly more GPU memory,
goal reward:
making it computationally inefficient for us.
Our student training objective linearly combines the RL (cid:16)
r =1 d <0.5∨
losses from PPO, the action distillation losses from [53], and move xy
arepresentationlossgivenbythemeansquarederrorbetween (cid:0) cosθ >0.5∧v ≤∥[v ] ∥≤v (cid:1)(cid:17) ,
vb,goal min b xy max
the teacher and student map embeddings. In practice, we also
(4)
disable the PPO surrogate loss during the first few iterations
while using a large learning rate. These design choices enable where θ vb,goal is the angle between the base velocity and the
stable, efficient, and well-aligned student training, which we vector from the base to the goal position, v min = 0.3m/s is
will ablate in Sec. VII. the lower bound of the horizontal base velocity [v b ] xy to be
considered as moving, and v = 2m/s is an upper bound
max
chosenaccordingtohardwareandSLAMconstraints.Inother
D. Environments
words,r =1iftherobotisalreadyclosetothegoalorifit
1) Rewards: We use three types of rewards shared across move
is moving roughly towards the goal, and r =0 otherwise.
both quadruped and biped training environments, as listed in move
Toenforceastablestandingpostureafterreachingthegoal,
Table I: task rewards which incentivize goal-reaching behav-
we define the standing reward:
iors,regularizationrewardswhichimprovestabilityandsafety,
andsimulation-fidelityrewards,whichpenalizenear-limitjoint r stand =1(d xy <0.5∧d yaw <0.5)
states that can cause unrealistic simulation. (cid:18) d +d +d +d (cid:19) (5)
·exp − foot g q xy ,
There are four task reward terms, respectively for goal 4
positiontracking,goalheadingtracking,movingtowardsgoal,
where d is the number of feet not in contact divided by
and standing at goal. Following [49], we define the position foot
the total number of feet, d = 1−[g ]2 measures the base
tracking reward as g b z
tilt relative to gravity, and d is the mean deviation of joint
q
1
r = ·t (4), (1) positionsfromthestandingreference.Thisrewardensuresthe
position tracking 1+0.25d2
xy
mask
robot maintains a static, upright configuration at the goal.
whered isthehorizontaldistancefromtherobottothegoal The regularization and simulation fidelity rewards are pre-
xy
position, and t (·) is a time-based mask function defined sented in Table I. We set all weights to integer powers of 10,
mask
as and make each term’s contribution 1∼2 orders of magnitude
1
smallerthanthetaskrewards.Thisenablessuccessfullearning
t (T)= ·1(t <T), (2)
mask T left without extensive weight tuning.
with t denoting the remaining time of the current episode. In the regularization rewards, we penalize each occurrence
left
Further, we define the heading tracking reward as of these undesired events:
1 • Spinningtoofast:Yawrate|[ω b ] z |>2.0rad/s,whichcan
r = ·t (2)·1(d <0.5), (3)
heading tracking 1+d2 mask xy trigger drifts in the odometry.
yaw

JOURNALSUBMISSION 7
| Training Terrains | ① Dense   ② Climbing   ③ Sparse |            |              |         |     |
| ----------------- | ------------------------------- | ---------- | ------------ | ------- | --- |
| Rough             |                                 | ① Stair Up | ① Stair Down | ① Boxes | ①   |
Obstacles ① Climbing Up ② Climbing Down ② Climbing Consecutive ②
| Gap |     | ③ Pallets | ③ Stones | ③ Beam | ③   |
| --- | --- | --------- | -------- | ------ | --- |
Test Terrains
Test 1: Irregular Stones ③ Test 2: Stones-Pallet-Climbing ②+③ Test 3: Parkour w/ Obstacles ①+②+③ Test 4: Debris ①+②+③
Fig.4. TrainingandTestTerrains.Wetrainourlocomotionpoliciesonprimitiveterrainsofthreecategories:⃝1 dense,⃝2 climbing,and⃝3 sparse.Totest
generalization,weevaluatethepoliciesonfourterrainsthatareeitherentirelyunseenorcombinationsoftrainingand/orunseenterrains.
• Leaping on flat terrain: All feet are off the ground when Unlike previous works [19], [23], [33], we do not explicitly
the elevation difference is <30 cm. reward or penalize foot contact positions in sparse terrains;
Non-foot contacts: We penalize both general non-foot instead, we formulate rewards for all robot links to enable
•
contacts and non-foot contact switches (from no contact emergent whole-body contacts. Those foothold position re-
tocontact).Weseparatethesetoallowstableinteractions wards are also difficult to define in a terrain-agnostic way:
when necessary (e.g., climbing) while discouraging un- for example, quadruped climbing benefits from active knee
necessary non-foot contacts. contacts and near-edge foot placements, as shown in Fig. 10,
Stumbling: Any link having a horizontal contact force whereas on sparse terrains near-edge foot placements are
•
| larger than   | the vertical | force.            | undesirable. |     |     |
| ------------- | ------------ | ----------------- | ------------ | --- | --- |
| Slippage: Any | link moving  | while in contact. |              |     |     |
•
Self-collision: Collisions between robot links. 2) Termination: To facilitate training, we set the following
•
|     |     |     | early termination | conditions: |     |
| --- | --- | --- | ----------------- | ----------- | --- |

| JOURNALSUBMISSION |              |        |               |          |         |            |            |     |     |     |     |     | 8   |
| ----------------- | ------------ | ------ | ------------- | -------- | ------- | ---------- | ---------- | --- | --- | --- | --- | --- | --- |
| Bad               | orientation: |        | The projected |          | gravity | vector     | satisfies  |     |     |     |     |     |     |
| •                 |              |        |               |          |         |            |            | (a) |     |     |     |     |     |
| |[g b             | ] x | >      | 0.985, | |[g b ] y     | | > 0.7, | or      | [g b ] z > | 0.0 (robot |     |     |     |     |     |     |
flipped).
| Base | collision: | The | base | link | experiences | a contact | force |     |     |     |     |     |     |
| ---- | ---------- | --- | ---- | ---- | ----------- | --------- | ----- | --- | --- | --- | --- | --- | --- |
•
| larger | than  | the robot’s   | total | weight. |              |     |           |     |     |     |     |     |     |
| ------ | ----- | ------------- | ----- | ------- | ------------ | --- | --------- | --- | --- | --- | --- | --- | --- |
| High   | thigh | acceleration: |       | The     | acceleration | of  | any thigh |     |     |     |     |     |     |
•
| link   | exceeds  | a      | specific  | threshold. |           | We design      | this | to  |     |     |     |     |     |
| ------ | -------- | ------ | --------- | ---------- | --------- | -------------- | ---- | --- | --- | --- | --- | --- | --- |
| reduce | impact   | during |           | jumping,   | as        | stiff landings | can  |     |     |     |     |     |     |
| damage | hardware |        | or reduce |            | lifespan. | Interestingly, | we   |     |     |     |     |     |     |
(b)
| use         | typical   | data     | from        | dog (60     | m/s2    | [71])     | and human     |     |     |     |     |     |     |
| ----------- | --------- | -------- | ----------- | ----------- | ------- | --------- | ------------- | --- | --- | --- | --- | --- | --- |
| (100        | m/s2      | [72])    | biokinetics | literature, |         | and       | they directly |     |     |     |     |     |     |
| work        | to emerge |          | impact      | absorption  | or      | avoidance | behav-        |     |     |     |     |     |     |
| iors,       | which     | is shown | in          | Sec. VI.    |         |           |               |     |     |     |     |     |     |
| Stagnation: |           | Movement |             | in the      | past 5s | is less   | than 0.5m     |     |     |     |     |     |     |
•
whiletherobotisstill>1mawayfromthegoalposition.
| Triggering  | any | of these | conditions |             | terminates |            | the episode |     |     |     |     |     |     |
| ----------- | --- | -------- | ---------- | ----------- | ---------- | ---------- | ----------- | --- | --- | --- | --- | --- | --- |
| immediately | and | incurs   | early      | termination |            | penalties. |             |     |     |     |     |     |     |
Fig.5. Thecontrollercanreusethebuiltmapforthesameterrain,enabled
byourmappingrandomization.(a)Whentherobotmovesinthedirectionof
| 3) Terrains |     | and Curriculum: |     | We  | train | our controllers | on  |     |     |     |     |     |     |
| ----------- | --- | --------------- | --- | --- | ----- | --------------- | --- | --- | --- | --- | --- | --- | --- |
thedottedorangearrow,itobservesapartialmapofthestaircase(visualized
primitive terrains and evaluate generalization on complex test ontheright).(b)Aftergoingdownalongthesolidorangearrow,itperceives
terrains.BothareshowninFig.4,whilequantitativeresultsare acompletemapofthestaircase(visualizedontheright)whenattemptingto
movebackalongthedottedorangearrow.
| presented  | in Sec.      | VII.     | We use | a terrain | curriculum  |             | that scales |                 |            |     |     |     |     |
| ---------- | ------------ | -------- | ------ | --------- | ----------- | ----------- | ----------- | --------------- | ---------- | --- | --- | --- | --- |
| difficulty | from         | easy to  | hard   | (detailed | in Appendix |             | A), similar |                 |            |     |     |     |     |
|            |              |          |        |           |             |             |             | E. Training and | Deployment |     |     |     |     |
| to [26].   | To stabilize | learning |        | across    | different   | challenging | ter-        |                 |            |     |     |     |     |
rains,weestimatetherobot’ssuccessrateusinganexponential 1) Training Setup: We train our controllers in Isaac
movingaverage.Iftherobotreachesthegoalanditsestimated Gym [61] with the PPO implementation in RSL-RL [74]. The
success rate exceeds 0.5, we promote the environment to the both teacher policies are trained with 80000 iterations, and
next difficulty level. Conversely, the level is demoted if the the both student policies are trained with 40000 iterations
robotremains>4mawayfromthegoal(whichistheaverage (surrogate loss disabled during first 5000 iterations). The
starting distance). Once the robot passes the highest level, the hyperparameters are listed in Appendix C. The training cost
environment resets to a random difficulty. for the ANYmal-D policies is ∼ 60 RTX-4090-days with 8
Additionally, we use a perception noise curriculum and an GPUs in parallel. The training cost for the TRON1 policies is
initialheadingcurriculum.Inthefirst20%iterationsofteacher ∼30 RTX-4090-days with 4 GPUs in parallel.
training, we linearly increase mapping noise from zero to the The difference between both robots’ training costs are due
maximumlevel,andsimultaneouslyexpandtheinitialheading to the mapping sizes. For ANYmal-D, we use a map of size
from facing the goal to a random yaw in [−π,π]. 36×14,with8-cmresolution,centeredatx=0.6 m,y=0 m
|                         |     |     |     |                          |     |     |     | in the base frame.    | For TRON1, | we        | use a map | of size | 18×13,   |
| ----------------------- | --- | --- | --- | ------------------------ | --- | --- | --- | --------------------- | ---------- | --------- | --------- | ------- | -------- |
| 4) DomainRandomization: |     |     |     | Weemploydomainrandomiza- |     |     |     |                       |            |           |           |         |          |
|                         |     |     |     |                          |     |     |     | with 8-cm resolution, | centered   | at x=0.32 |           | m,y=0   | m in the |
tion [73] during training to enhance robustness and facilitate base frame. These numbers are designed based on the robot
sim-to-real transfer. We apply the following randomization dimensions and terrain sizes.
setup:
Robot Dynamics: We uniformly randomize the payload, 2) Deployment Setup: We deploy the controllers using
•
frictioncoefficients,andactuationdelays.FortheTRON1
|     |     |     |     |     |     |     |     | ONNX Runtime | [75]. On | the onboard | Intel | Core | i7-8850H |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | -------- | ----------- | ----- | ---- | -------- |
biped, we additionally randomize PD gains and motor CPUs, the policy inference time is approximately 2 ms.
| armatures, |     | following | [15]. |     |     |     |     |     |     |     |     |     |     |
| ---------- | --- | --------- | ----- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
• ObservationNoise:Weapplyuniformnoisestothepolicy 3) Other Sim-to-Real Designs: We model the actuator
observations. We also degrade the student’s depth clouds dynamics for sim-to-real transfer. In simulation, we use the
| by  | simulating | missing | points |     | and sensor | artifacts. |     |                  |                    |     |     |               |     |
| --- | ---------- | ------- | ------ | --- | ---------- | ---------- | --- | ---------------- | ------------------ | --- | --- | ------------- | --- |
|     |            |         |        |     |            |            |     | actuator network | [76] for ANYmal-D, |     | and | an identified | DC- |
• Mapping: During student training, we randomly select a motor model for TRON1. We clip the applied torques using
subset of environments to access complete maps, while the joint torque-velocity constraints in [77].
others rely on partial online maps. This setup enables To enable continuous deployment (infinite-horizon execu-
map reuse when the robot traverses the same terrain tion), we define the command observations differently from
| repeatedly, |     | as illustrated |     | in Fig. | 5. We | also | corrupt the |                     |                           |     |     |       |           |
| ----------- | --- | -------------- | --- | ------- | ----- | ---- | ----------- | ------------------- | ------------------------- | --- | --- | ----- | --------- |
|             |     |                |     |         |       |      |             | prior goal-reaching | works [48]–[50].Following |     |     | prior | practice, |
student’s mapping by randomly removing points and the critic receives the full command: the goal’s relative posi-
assigning random height values with high uncertainties. tion,thesineandcosineoftherelativeyaw,andtheremaining
Additionally, we simulate mapping drifts for both the episode time. The actor, however, receives a modified com-
teacher and the student. mand representation: we clip the observed goal distance to a
Implementation details are provided in Appendix B. maximumof2mandremovetheremainingtime.Additionally,

JOURNALSUBMISSION 9
Odometry Pose
Global Mapping Local Mapping
Depth Cloud Local Grids Predictions and Uncertainties (High-Uncertainty Points Omitted) (Used by Controller)
Fuse Query
Projection Neural Network
Fig.6. Ourproposedmappingpipeline.Foreachframeofdepthclouds,weprojectitintolocalgrids,andusealightweightneuralnetworktopredictelevation
estimationswithuncertainties.Thepredictionsarethenfusedintotheglobalmapviatheodometry,andlocalmapscanbequeriedfromtheglobalmapto
serveasthecontrollerinputs.Invisualization,thegreenpointsindicatethedepthpointsorelevationestimations,andthebluelinesindicatetheuncertainties.
Locomotion Training Terrains Stacked Boxes layer and the uncertainty layer (in the form of variance).
The global map is initialized as flat ground at the robot’s
standing height (ground height relative to the base) with large
uncertainties. At each new frame, given the local elevation
estimations with uncertainties and the current base pose, we
project the local predictions to the global grid cells. For any
global grid point (u,v) covered by the local grids, we denote
thenewestimationash anditsuncertaintyasσ2,fusingthem
t t
Random Heightfields Floating Boxes with existing values h and σ2 .
prior prior
We do not use standard Bayesian fusion because repeated
observationsofthesameoccludedoruncertainareashouldnot
havereduceuncertaintiessimplyduetoconsistentpredictions.
Instead, we employ a Probabilistic Winner-Take-All strategy.
First, we calculate an effective measurement variance σˆ2 that
t
is lower-bounded by the prior to prevent over-confidence:
Fig.7. Terrainmeshesusedformappingmodeltraining.Weusefourmeshes: σˆ2 =max(σ2,0.5·σ2 ). (6)
t t prior
thelocomotiontrainingterrains,randomlystackedboxes,randomheightfields,
and random floating boxes. The colored points visualized are the elevation
An update is considered valid only if the effective measure-
scansfromrandomposes.
ment variance is not significantly larger than the prior (σˆ2 <
t
if the actual goal distance exceeds 2m, we randomize the 1.5σ2 ), or if the absolute uncertainty is low (σˆ2 <0.22).
prior t
observedyawcommandduringtraining.Thisdesigndecouples
For valid updates, we determine the probability p of
win
policy deployment from finite-horizon training and simplifies
overwriting the map based on the relative precision:
steering control.
(σˆ2)−1
p = t . (7)
V. NEURALMAPPING win (σˆ t 2)−1+(σ p 2 rior )−1
A. Mapping Pipeline Finally, the map is updated stochastically. We sample ξ ∼
Our proposed neural mapping pipeline is illustrated in U[0,1] and let the new prediction take over the cell if the
Fig. 6. It can not only achieve real-time computation on the sample falls within the probability threshold:
hardware,butalsorunwiththousandsofparallelenvironments (cid:40)
(h ,σˆ2) if ξ <p ,
in simulation, thereby bridging the sim-to-real gap with iden- (h ,σ2 )← t t win (8)
tical mapping pipelines. new new (h prior ,σ p 2 rior ) otherwise.
For each sensing frame, we project the point cloud onto
For the controller inputs, we then just query the grids around
a local 2D height grid. If multiple points fall into the same
the robot pose in the latest global map.
cell, we keep the maximum z-value, which is most relevant
This Probabilistic Winner-Take-All strategy offers the fol-
for locomotion, and we assign a fixed minimum value to cells
lowing benefits:
with no points. The resulting local elevations, however, are
often noisy and incomplete due to occlusions. To mitigate • The uncertainty of the same occluded point will not
this,weusealightweightCNNtrainedwithBayesianlearning decrease through consistent predictions.
(detailedinSec.V-B)tojointlypredictbase-relativeelevations • Over-confident predictions, if not consistent, cannot take
and their uncertainties (in the form of log-variance). The over the cell.
predicted uncertainties capture both measurement noise and • The system can rapidly update the map in response to
occlusions,whiletheelevationpredictionsthemselvescanalso dynamic terrain changes when high-confidence measure-
suppress noise. ments are available.
We then use the odometry poses to fuse these local predic- • The pipeline is easy to integrate with parallel simulation,
tionsintoaglobalgridmapM.Ithastwolayers:theelevation and fast enough to run on the hardware.

JOURNALSUBMISSION 10
2× CNN 2× CNN
CNN
Input channel=16 channel=16 Uncertainty Output
Concat channel=1
kernel_size=3 kernel_size=3
kernel_size=1
padding=1 padding=1
CNN
Raw Estimation
channel=1
MaxPool
kernel_size=1
kernel_size=3 Upsample
stride=3
CNN
Gating Estimation Output
channel=1 Sigmoid
kernel_size=1
Fig.8. Mappingmodelarchitecture.WeusealightweightU-Net[78]modelwithagatedresidualdesignfortheestimation.
B. Training cases. To mitigate this, we re-weigh samples in each batch
during training using their total variation (TV) [81]:
We train a CNN to predict per-frame elevation estimates
anduncertaintieswithsyntheticdataandrandomterrains.The 1 (cid:16) (cid:17)
TV(Y )= ∥∇ Y ∥ +∥∇ Y ∥ ,
technical implementations are detailed below. b HW x b 1 y b 1
(10)
1) Terrains and Data Sampling: We use both the loco- w = TV(Y b ) .
motion training terrain meshes and additional procedurally b (cid:80)B TV(Y )+ε
b′=1 b′
generated terrains to train the mapping model, as shown in
Here, Y is the ground-truth elevations of the b-th sample in
Fig. 7. Instead of relying on physical simulation, we directly b
a batch, and H and W are the height and width. The weight
sample local elevation grids from random poses above these
w is normalized across the batch, and ε is a small positive
meshesusingraytracingwithWarp[79],enablingsamplingat b
constant. By doing so, we assign higher weights to samples
hundreds of thousands of frames per second on a single RTX
with larger elevation variations.
4090 GPU.
We use a shallow U-Net [78] model with a gated residual
2) Training Data Synthesis: We apply the following aug-
design, as illustrated in Fig. 8. The CNNs output the uncer-
mentations to the sampled local elevation grids:
tainty,arawestimation,andagatingmap.Thefinalestimation
• additive uniform noise with random magnitude on each isobtainedbythegatedcombinationoftherawestimationand
cell;
the input, preserving accuracy in clearly observed areas while
• random cropping of the map from the four borders; selectively overwriting noisy or occluded areas.
• simulated occlusions with random sensor positions and We train the models on 54 million frames for each robot,
field of view;
each model takes less than 1 hour to converge. For ANYmal-
• random ranges to clip the elevations; D, the local grids are of shape 51×31, with 4-cm resolution,
• random missing points and outliers at random ratios. centered at x = 1.0 m,y = 0 m in the base frame. For
By doing so, we synthesize diverse, noisy, and partially TRON1, the local grids are of shape 31 × 31, with 4-cm
observable local grids as the model inputs, and the original resolution, centered at x=0.6 m,y=0 m in the base frame.
ground-truth elevations as the labels.
3) Model and Optimization: We train the model to recon- C. Simulation Integration and Deployment
struct the ground-truth elevations with β−NLL loss (β =0.5)
Whendeployedinsimulationwith1000parallelANYmal-D
from [80]:
environments,themodel’sinferencetimeisbelow0.3ms,and
(cid:20) (cid:18) logσˆ2(X) (Y −µˆ(X))2(cid:19)(cid:21) the GPU memory consumption is about 3 GB (for TRON1,
L =E sg[σˆ(X)] + ,
0.5 X,Y 2 2σˆ2(X) these numbers are roughly 60% of those for ANYmal-D).
(9) Storing 8,m×8,m global maps for 1000 environments re-
where X denotes the inputs, Y denotes the ground-truth ele- quires about 0.3 GB, and the global map size can be flexibly
vations,andµˆ(X)andσˆ2(X)denotethepredictedestimation chosen since we recenter the map around the robot when it
andvariance,respectively.Theoperatorsg[·]denotesthestop- approachestheboundary.Obtainingdepthcloudsandhandling
gradient operation. Compared to the standard negative log- other intermediate overheads takes an additional ∼1.2 GB of
likelihood(NLL)lossusedinclassicalBayesianlearning[51], GPU memory per 1000 ANYmal-D environments.
thisformulationreducesthetendencyofthemodeltooveresti- Onrealhardware,themappingpipelinetakesapproximately
mateuncertaintyonhardsamplestotriviallyreducetheloss.It 5msperframeontheonboardCPU,witharound2.5msspent
encouragesthemodeltooutputhighuncertaintywhenaccurate on model inference using ONNX Runtime. This enables fast,
predictionsarenotpossible,andlowuncertaintytogetherwith low-latencymappingthatcankeepupwiththedepthcamera’s
accuratepredictionswhentheyare,therebycapturingthenoise frame rate. For ANYmal-D, we merge the point clouds from
and occlusions. two front facing cameras for processing. For TRON1, we use
Forbatchedoptimization,terrainroughnesscanvarysignif- the only front facing camera.
icantly across samples. As a result, flat terrains can dominate Regardingodometry,weuseCompSLAM[82]withGraph-
thebatchlossandreducetheeffectiveemphasisonchallenging MSF [83] on ANYmal-D to obtain a high-frequency, accurate

JOURNALSUBMISSION 11
1) Quadruped Parkour: We demonstrate the agility and
generalization of our trained ANYmal-D controller on a park-
ourcourse,asshowninFig.10.Thiscourseisnotincludedin
the training terrains of either the locomotion controller or the
mappingmodel,yetoursystemcanstablytraverseitatspeeds
of up to 2 m/s, composing climbing and jumping maneuvers.
2) Sparse Terrain Locomotion: As shown in Fig. 11, our
controllers enable both ANYmal-D and TRON1 to traverse a
variety of sparse terrains, many of which are unseen during
training, exhibiting generalization capabilities.
3) Bipedmaneuvering: Todemonstratetheomnidirectional
perceptive locomotion capabilities of our controller, we com-
mand TRON1 to traverse a 38-cm-high platform, a gap, a
staircase, and rough terrain, as shown in Fig. 12. The robot
executes smooth maneuvers in all directions.
4) Robustness: We demonstrate the robustness of our con-
trollers on moving terrains. As shown in Fig. 13, TRON1 can
traverse or balance on a platform cart with unlocked wheels
and spring suspension that can easily tilt to one side with
the robot weight. We also show that ANYmal-D can recover
from unfixed tilted stepping stones using its knees, after an
occasional trip causes a misstep that tilts the blocks.
5) Mapping Results: In Fig. 14, we show the mapping re-
sultsafterANYmal-Dtraversesterrainsthatareunseenduring
training. These maps capture fine-grained terrain details, such
Fig.9. OurANYmal-Dcontrollerzero-shotsthehardestparkourandrubble as gaps and supports, that help the controller achieve agile
pileterrainsreportedinpriorwork[32],[48].
and generalized locomotion. Among prior works, only [32]
LiDAR–inertial odometry. On TRON1, we use DLIO [84] has demonstrated maps of comparable quality during agile
instead,asitbetterhandlestheplatform’shigheraccelerations. motions. However, that approach cannot reliably infer unseen
Notably, although these odometry solutions can run reliably parts of obstacles in unstructured terrains [48], is not efficient
at high frequency, their velocity estimates are observed to enough to support concurrent locomotion training, and does
be noisy and delayed. This is intuitive: for example, a 1 cm not explicitly encode uncertainty under partial observations
positiondriftovera20mstimestepcantranslatetoavelocity such as occlusions. For generalized locomotion, explicitly
errorof0.5m/s,yethaslittleimpactonmapfusion.Therefore, modeling uncertainty is important: occluded regions are not
we remove the linear velocity observations from the student completed by learned priors but kept uncertain, and newly
policy. observed geometry can be integrated into the map based on
the predicted uncertainty once it becomes visible.
VI. RESULTS
C. Emergent Behaviors
A. Agility Comparison with Prior Art
1) ActivePerception: Trainedwithpartialobservationsand
We achieve state-of-the-art agility on both ANYmal-D and
uncertainty-awaremapping,ourlocomotioncontrollersexhibit
TRON1, measured by the difficulty of the terrains they can
emergentactiveperceptionbehaviors.InFig.15,weexemplify
climbupanddown.OnANYmal-D,despitebeingtrainedonly
thiswithANYmal-Dattemptingtoclimbontoahighobstacle.
onprimitiveterrains,ourpolicyzero-shotsthehardestparkour
In one attempt, the robot makes contact with the obstacle
and rubble pile terrains reported in prior work [32], [48], as
and cannot immediately climb up, due to its limited field of
shown in Fig. 9. On TRON1, our policy climbs up platforms
view and imperfect depth sensing. This interaction, however,
upto0.48mandclimbsdownupto0.88m,whilepriorwork
exposes higher parts of the obstacle to the map. In the
on bipeds reports 0.5 m platforms [17] using the Unitree H1
subsequent attempt, the controller uses this updated map and
robot (with 4× peak torque and 1.5× base height compared
successfullyclimbsontotheobstacle.Themapsarevisualized
to TRON1).
to show how information gained from previous interactions is
Acrossallterrains,ourcontrollersonbothrobotsreachpeak
reused to make subsequent movements more informed and
forwardvelocitiesabove1.5m/s,demonstratingacombination
successful.
of agility and terrain-aware locomotion. On top of this agility,
Notably, this addresses a stated limitation of prior state-of-
our controllers also exhibit stronger generalization than exist-
the-art work [48], where a recurrent controller fails to climb
ing methods, as shown below and benchmarked in Sec. VII.
onto a high box when starting close to it due to a lack of
longer-term memory. In contrast, our mapping module ex-
B. Real World Results plicitly maintains long-term spatial information, providing the

| JOURNALSUBMISSION |     |     |     |     |     |     |     |     |     | 12  |
| ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
Fig.10. OursystemenablesANYmal-Dtomovebackandforthonaparkourcourse,aterrainthatisunseenduringtrainingofboththecontrollerandthe
mappingmodel.
| (a) |     |     |     | (b) |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (c) |     |     |     | (d) |     |     |     |     |     |     |
| (e) |     |     |     | (f) |     |     |     |     |     |     |
Fig.11. OursystemsenableANYmal-DandTRON1totraversediversesparseterrains.(a)ANYmal-Dtraversesa19-cmwidebalancebeam,whichisin
thetrainingterrains.(b)TRON1traversestwounfixedfloating19-cmwideblocks,whichmakeanunseencurvedbeamterrain.(c)ANYmal-Dtraversestwo
unfixedbeamswithgaps,whichisunseenduringtraining.(d)TRON1traversesabeamfollowedbyagap,whichisanunseencombinationduringtraining.
(e) ANYmal-D traverses two rows of 19-cm wide stepping stones with 10-cm height differences, which is unseen during training. (f) TRON1 traverses
diamond-layoutsteppingstones,whichisunseenduringtraining.
operationbyreducingtheneedforhigh-frequency,navigation-
|     |     |     |     | level commands | (such             | as velocity   | commands      |            | [13]).      |               |
| --- | --- | --- | --- | -------------- | ----------------- | ------------- | ------------- | ---------- | ----------- | ------------- |
|     |     |     |     | 3) Whole-Body  | Contact:          |               | As previously |            | shown       | in Fig. 10    |
|     |     |     |     | and Fig.       | 13, our ANYmal-D  |               | controller    |            | actively    | use knee      |
|     |     |     |     | contacts       | to help stabilize | the           | robot         | and        | traverse    | challenging   |
|     |     |     |     | terrains.      | On one hand,      | such          | whole-body    |            | contact     | behaviors     |
|     |     |     |     | are beneficial | for agility       | and           | robustness;   |            | on the      | other hand,   |
|     |     |     |     | because        | most existing     | legged        | robots        | are        | designed    | primarily     |
|     |     |     |     | for foot       | contacts, these   | behaviors     |               | may stress |             | the hardware. |
|     |     |     |     | That said,     | we believe        | that          | whole-body    | contact    |             | will become   |
|     |     |     |     | increasingly   | important         | for achieving |               | whole-body |             | dexterity in  |
|     |     |     |     | complex        | environments      | [64],         | [85], [86].   |            |             |               |
|     |     |     |     | 4) Impact      | Reduction:        | As            | mentioned     | in         | Sec. IV-D2, | we use        |
Fig.12. TRON1maneuversovera38-cm-highplatform,agap,astaircase, human and dog motion data to set thigh-acceleration thresh-
| and rough | terrain, showcasing | its omnidirectional | perceptive locomotion |     |     |     |     |     |     |     |
| --------- | ------------------- | ------------------- | --------------------- | --- | --- | --- | --- | --- | --- | --- |
capabilities.Orangearrowsindicatetherobot’sapproximatetrajectories. olds for early termination. This induces impact-reduction be-
|     |     |     |     | haviors,   | as shown in     | Fig. 17: | when  | climbing | down, | ANYmal-  |
| --- | --- | --- | --- | ---------- | --------------- | -------- | ----- | -------- | ----- | -------- |
|     |     |     |     | D uses its | knees to gently | touch    | down, | while    | TRON1 | retracts |
controller with an effective representation of the environment the support leg to absorb the impact.
overtime.Moreover,consistentwith[48],wealsoobservethe
| robot looking | upward while         | climbing onto | platforms, gaining |                  |         |          |     |     |     |     |
| ------------- | -------------------- | ------------- | ------------------ | ---------------- | ------- | -------- | --- | --- | --- | --- |
|               |                      |               |                    | D. Interpretable | Feature | Patterns |     |     |     |     |
| additional    | terrain information. |               |                    |                  |         |          |     |     |     |     |
WevisualizethefeaturepatternsofourANYmal-Dstudent
2) Loco-Navigation: Since our locomotion controllers are policy’s AME-2 encoder in Fig. 18. For local features, we
trained to reach goals, they also exhibit local navigation visualize their attention weights with a red–blue colormap.
capabilities. As shown in Fig. 16, the controllers can align For global features, which are obtained via max pooling, we
the robot with the terrain before traversal and avoid collisions treat pooled points as selected (weight 1) or not (weight 0)
with obstacles. These behaviors can simplify remote human and average this binary mask over the feature dimension.

JOURNALSUBMISSION 13
(a) (a) (b) (c)
(b)
Fig. 15. Emergent active perception behavior. (a) In a climb-up attempt,
ANYmal-Dcollideswiththeobstacleduetolimitedfieldofviewanddepth
sensing quality. (b) Although unsuccessful, this trial reveals higher parts of
theobstacleinthemap,enablingtherobottosucceedintheimmediateretrial.
(c)ANYmal-Dthenclimbsontotheobstacle.Themapsarevisualizedinthe
top-leftcornerofeachsubfigure,excludingregionsthatarenevercoveredby
thecameras.
(c) (a)
(b)
Fig. 13. Our controllers are robust to moving terrains. (a) TRON1 climbs
overanunlockedtiltableplatformcart.(b)TRON1climbsontoandbalances
on an unlocked tiltable platform cart. (c) ANYmal-D recovers from unfixed
tiltedsteppingstonesusingitsknees.
Fig.16. Emergentlocalnavigationcapabilities.(a)TRON1turnsaroundand
aligns itself with the beam to traverse the terrain. (b) ANYmal-D avoids an
obstaclewhenreachingthegoal.
Terrains Unseen During Training Mapping After Traversal
VII. ABLATIONSANDBENCHMARKING
In this section, we benchmark our design choices on test
terrains (depicted in Fig. 4) that are unseen during training.
All benchmarking experiments are conducted on ANYmal-
D in simulation, where richer prior works and baselines are
available.
A. Teacher Architecture
We compare our teacher policy architecture against two
baselines:AME-1[15]andMoE[68].AME-1isthepriorstate
of the art in generalized locomotion, outperforming policies
based on MLPs, CNNs, multimodal transformers [37], and
Vision Transformers [87]. The MoE architecture from [68]
has demonstrated strong scalability by enabling a humanoid
to track diverse human motions.
Fig.14. ObtainedmapsafterANYmal-D’straversalofterrainsthatareunseen
The results are summarized in Table II. All architecture
during training. Our mapping supports agile and generalized locomotion. In
the visualizations, meshes represent the estimated elevations, and thin lines designs scale well on the training terrains, highlighting the
indicatetheuncertainties.Themeshesarecoloredwithared–bluecolormap stability of our training framework. However, MoE shows
to enhance height contrast, excluding regions that are never covered by the
very limited generalization to unseen test terrains. AME-1
cameras.
generalizes well on sparse terrains (test 1) but struggles when
different terrains are mixed (test 2-4), which we attribute to
its encoder assigning attention to local features based only on
By inspecting these feature maps, we observe that local
proprioception, thus missing global context. In contrast, our
attention weights often emphasize fine-grained terrain details,
AME-2 teacher generalizes reliably across all test terrains.
while global features tend to concentrate on a sparse set of
distinctive points across terrain types (e.g., obstacle bound-
B. Student Designs
aries, high platform surfaces, and beam centers). We also find
that these patterns persist on test terrains, which helps explain Having a generalized teacher does not automatically yield
ourcontroller’sgeneralizationtounseenterrainsdespitebeing a generalized deployable student. We therefore compare our
trained only on primitive terrains. AME-2 student against the following baselines:

| JOURNALSUBMISSION |     |     |     |     |     |     |     |     |     |     |     |     |     |     | 14  |
| ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
TABLEII
SUCCESSRATES(%)OFDIFFERENTTEACHERARCHITECTURESONTRAININGANDTESTTERRAINS
|     |     |     |     |     | TrainingTerrains |     |     |     |     | TestTerrains |     |     |     |     |     |
| --- | --- | --- | --- | --- | ---------------- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- | --- |
TeacherArchitecture
|     |     |                |     | Dense | Climbing |      | Sparse | Test1 | Test2 | Test3 | Test4 | avg.Test |      |     |     |
| --- | --- | -------------- | --- | ----- | -------- | ---- | ------ | ----- | ----- | ----- | ----- | -------- | ---- | --- | --- |
|     |     | AME-2(Teacher) |     | 97.9  |          | 96.0 | 95.2   | 96.8  | 93.8  | 99.7  |       | 90.6     | 95.2 |     |     |
|     |     | AME-1[15]      |     | 96.0  |          | 95.7 | 95.5   | 99.2  | 29.2  | 32.2  |       | 44.1     | 51.2 |     |     |
|     |     | MoE[68]        |     | 95.0  |          | 96.7 | 90.5   | 52.9  | 55.3  | 41.0  |       | 30.8     | 45.0 |     |     |
Bestresults(within5%ofthemaximum)areshowninbold.
TABLEIII
SUCCESSRATES(%)OFDIFFERENTSTUDENTDESIGNSONTRAININGANDTESTTERRAINS
|     |     |     |     |     | TrainingTerrains |     |     |     |     |     | TestTerrains |     |     |     |     |
| --- | --- | --- | --- | --- | ---------------- | --- | --- | --- | --- | --- | ------------ | --- | --- | --- | --- |
StudentDesign
|     |     |                |     |     | Dense | Climbing | Sparse |     | Test1 | Test2 | Test3 | Test4 | avg.Test |     |     |
| --- | --- | -------------- | --- | --- | ----- | -------- | ------ | --- | ----- | ----- | ----- | ----- | -------- | --- | --- |
|     |     | AME-2(Student) |     |     | 96.4  | 96.4     | 91.8   |     | 90.0  | 77.7  | 89.1  | 72.9  | 82.4     |     |     |
VisualRecurrentStudent[48] 98.0 97.9 86.4 34.5 20.9 99.8 50.6 51.5
|     | AME-2(Student)w/oRL |     |     |     | 95.8 | 87.9 | 84.6 |     | 46.9 | 59.1 | 90.6 | 46.0 | 60.7 |     |     |
| --- | ------------------- | --- | --- | --- | ---- | ---- | ---- | --- | ---- | ---- | ---- | ---- | ---- | --- | --- |
AME-2(Student)w/orep.loss 95.8 94.8 86.5 81.9 82.6 69.5 60.4 73.6
|     |     | AME-2(Teacher) |     |     | 97.9 | 96.0 | 95.2 |     | 96.8 | 93.8 | 99.7 | 90.6 | 95.2 |     |     |
| --- | --- | -------------- | --- | --- | ---- | ---- | ---- | --- | ---- | ---- | ---- | ---- | ---- | --- | --- |
Beststudentresults(within5%ofthemaximum)areshowninbold.
(a) Regarding the learning setup, we find that direct teacher
|     |     |     |     |     |     |     | supervision |          | (action           | distillation     |                    | plus representation |          | alignment)      |           |
| --- | --- | --- | --- | --- | --- | --- | ----------- | -------- | ----------------- | ---------------- | ------------------ | ------------------- | -------- | --------------- | --------- |
|     |     |     |     |     |     |     | already     |          | yields            | good performance |                    | on the              | training | terrains,       | but       |
|     |     |     |     |     |     |     | does        | not      | produce           | strong           | generalization     |                     | to       | unseen          | terrains  |
|     |     |     |     |     |     |     | without     |          | RL. Incorporating |                  | RL                 | is important        |          | for performance |           |
|     |     |     |     |     |     |     | on          | the test | terrains,         | and              | the representation |                     | loss     | gives           | a further |
(b)
|     |     |     |     |     |     |     | improvement. |        |         | Overall,    | our student  | policy  |                | architecture | per-   |
| --- | --- | --- | --- | --- | --- | --- | ------------ | ------ | ------- | ----------- | ------------ | ------- | -------------- | ------------ | ------ |
|     |     |     |     |     |     |     | forms        | better | on      | unseen      | terrains     | than    | the end-to-end |              | visual |
|     |     |     |     |     |     |     | recurrent    |        | policy, | and         | our learning | design, |                | with mixed   | RL     |
|     |     |     |     |     |     |     | losses       | and    | teacher | supervision |              | losses  | (action        | distillation | and    |
representationalignment),furtherenhancesthegeneralization.
Fig.17. Emergentimpactreductionbehaviors.(a)ANYmal-Dusesitsknee
tosupportthebodywhenclimbingdown,andgentlytouchesthegroundto
reduce the impact. (b) TRON1 extends one leg while jumping down, and C. Mapping Designs
retractsitduringlandingbeforeswitchingsupporttotheotherleg.
|     |     |     |     |     |     |     |     | We benchmark |     | our mapping |     | pipeline | against | two baselines: |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------ | --- | ----------- | --- | -------- | ------- | -------------- | --- |
1) The end-to-end student design from [48], the previous 1) Our pipeline, with the mapping model trained only on
stateoftheartinagilegeneralistlocomotion,whichuses locomotion training terrains.
recurrent networks with vision inputs. 2) A temporal recurrent model that directly predicts ego-
2) AnAME-2studenttrainedwithoutRLlosses,usingonly centric elevation maps from local grid observations and
| teacher     | action | distillation    | and     | the representation |                    | loss. |     | delta | transforms. |       |         |            |     |         |        |
| ----------- | ------ | --------------- | ------- | ------------------ | ------------------ | ----- | --- | ----- | ----------- | ----- | ------- | ---------- | --- | ------- | ------ |
| 3) An AME-2 |        | student trained | without |                    | the representation |       |     |       |             |       |         |            |     |         |        |
|             |        |                 |         |                    |                    |       | The | first | baseline    | tests | whether | additional |     | diverse | meshes |
loss.
|     |     |     |     |     |     |     | (both | traversable |     | and untraversable) |     | are | necessary | for | better |
| --- | --- | --- | --- | --- | --- | --- | ----- | ----------- | --- | ------------------ | --- | --- | --------- | --- | ------ |
The results are summarized in Table III. On the training generalization. The second contrasts our per-frame fusion
terrains, all student policies perform well, with our proposed approach with an end-to-end recurrent neural memory model:
design achieving slightly higher success rates on sparse ter- therecurrentmodeltendstooverfitterrainpatternsseenduring
rains. On unseen test terrains, our student achieves the best training and produces lower-quality maps on unseen terrains,
overall success rates. as we will show in the qualitative comparison in Fig. 19.
The end-to-end visual recurrent student from [48] performs All pipelines are evaluated on data collected from the same
slightly better than ours on Test 3, a parkour course with rollout trajectories, with results reported in Table IV. We use
obstacles, but underperforms on all other test terrains that theL lossdefinedinEq.9asthemetric,whichbalanceses-
0.5
containsparseregions.OnTest3,mostofourstudent’sfailures timationaccuracyanduncertaintyquantification.Thetemporal
occur when it attempts to climb the obstacles, which can be recurrentmodel(detailedinAppendixD)usesaCNNencoder
misinterpreted as higher platforms due to occlusions and the for local grids, an MLP encoder for delta transforms, and
observed surface elevations. In contrast, using the full depth Spatially-Enhanced Recurrent Units (SRU), an architecture
image, as in the visual recurrent design, may provide richer thatenables100m-scalemaplessvisualnavigation[88],asthe
cues and a longer reaction horizon for obstacle avoidance. memory module. To stabilize training and prevent the model
Despite this modest drop in generalization around obstacles, from collapsing to large uncertainties and poor estimates, we
our student controller achieves much higher success rates on have to train the recurrent model only on locomotion training
sparse and mixed terrains. terrainswiththepolicyintheloop(asdoneinpriorwork[29],

| JOURNALSUBMISSION |            |     |           |     |             | 15  |
| ----------------- | ---------- | --- | --------- | --- | ----------- | --- |
|                   | Stair Down |     | Obstacles |     | Climbing Up |     |
Local Attention Global Context Local Attention Global Context Local Attention Global Context
|     | Climbing Down |     | Gap |     | Beam |     |
| --- | ------------- | --- | --- | --- | ---- | --- |
Local Attention Global Context Local Attention Global Context Local Attention Global Context
|     | Test 1 |     |     | Test 2 |     |     |
| --- | ------ | --- | --- | ------ | --- | --- |
Local Attention Global Context Local Attention Global Context Local Attention Global Context
|     | Test 3 |     |     | Test 4 |     |     |
| --- | ------ | --- | --- | ------ | --- | --- |
Local Attention Global Context Local Attention Global Context Local Attention Global Context
Fig.18. FeaturepatternsofourANYmal-Dcontroller.Wevisualizetheneuralelevationmaptogetherwiththecorrespondinglocal-attentionandglobal-context
weights.Pointsindicateelevationestimates,linesindicateuncertainties,andhigher-intensityredcolorscorrespondtohigherweights.Localattentionweights
providefine-grainedpatternsforlocomotioncontrol,whileglobalcontextfeaturesprimarilyfocusonsparsepointsthatcharacterizetheterraintype.
[33])andaddanauxiliaryL2lossontheelevationestimatesin (replaced by random points in depth clouds), and with the
additiontoL .Thissetuprequiresroughly5×moretraining front upper camera disabled. The tested setups are all above
0.5
time to converge than our mapping models. the randomization ranges during training.
It is worth noting that the recurrent model can optimize The results are presented in Table V. Under missing points
both the elevation estimates and the associated uncertainties and artifacts, we find the policy performance does not drop,
in unseen regions based on the training distribution, which and can sometimes be slightly better. This can be explained
is advantageous under the L loss, whereas our pipeline by that, fewer effective points in the depth clouds result in
0.5
uses heuristic defaults. However, as illustrated in Fig. 19, our more uncertain regions and make the policy behave more
| mapping | pipeline produces higher-quality | maps in practice, | conservatively. |     |     |     |
| ------- | -------------------------------- | ----------------- | --------------- | --- | --- | --- |
and can effectively model occlusions with high uncertainties. When the front upper camera is disabled, the system still
|     |     |     | performs well | on most terrains | without re-training, | highlight- |
| --- | --- | --- | ------------- | ---------------- | -------------------- | ---------- |
ingtheflexibilityofourmethodfordifferentsensorconfigura-
| D. Robustness | to Visual Noise |     |     |     |     |     |
| ------------- | --------------- | --- | --- | --- | --- | --- |
tions.However,onterrainsthatrequireactiveperception,such
Wevalidatetherobustnessofoursystemtonoiseandsensor as the high obstacles to climb in Test 2 and Test 3, the policy
degradation. We evaluate our student policy’s success rates struggles, suggesting that these behaviors remain sensitive to
under missing points (removed from depth clouds), artifacts the available sensor viewpoints.

JOURNALSUBMISSION 16
TABLEIV
L0.5LOSSOFDIFFERENTNEURALMAPPINGMETHODSONTRAININGANDTESTTERRAINS
TrainingTerrains TestTerrains
MappingPipeline
Dense Climbing Sparse Test1 Test2 Test3 Test4 avg.Test
Ours -0.006 -0.108 0.020 0.033 0.227 0.036 -0.111 0.046
Ours(Loco-Only) 0.020 -0.090 0.081 0.104 0.234 0.087 -0.075 0.088
TemporalRecurrent -0.162 -0.114 -0.101 0.316 0.135 0.066 -0.176 0.085
Best(lowest)resultsareshowninbold.
TABLEV
SUCCESSRATES(%)OFTHESTUDENTPOLICYUNDERNOISEANDSENSORDEGRADATION
TrainingTerrains TestTerrains
Setup
Dense Climbing Sparse Test1 Test2 Test3 Test4 avg.Test
Nominal 96.4 96.4 91.8 90.0 77.7 89.1 72.9 82.4
20%MissingPoints 95.5 94.9 91.3 91.1 77.3 90.3 74.8 83.4
3%Artifacts 96.4 97.6 93.2 95.2 72.8 92.5 80.3 85.2
1CameraDisabled 91.2 82.0 85.9 95.1 56.0 1.1 74.6 56.7
VIII. DISCUSSION drawback is that it offers less direct control over the robot’s
intermediatemotion,whichcanbetroublesomewhensemantic
In this paper, we propose AME-2, a unified RL framework
obstacles lie along the path but appear traversable from the
for agile and generalizable legged locomotion with attention-
geometry. We envision combining both command interfaces
based neural map encoding. We use goal-reaching rewards to
within a single controller in future work, e.g., via masked
incentivize agile behaviors, and design the AME-2 encoder
commands and distillation [89].
to achieve generalization across diverse challenging terrains.
To facilitate sim-to-real deployment, we combine teacher-
student learning with RL and introduce a lightweight neural C. Whole-Body Skills
mapping pipeline that explicitly models partial observability
We achieve whole-body locomotion skills in this work, but
whilerunningidenticallyinsimulationandonrealrobots.Our
for a given terrain type the learned motions tend to follow
framework applies to both a quadruped and a biped without
similar contact patterns. For systems with higher DoFs, such
changing the rewards or training settings.
as humanoids, the same terrain type at different scales may
require different contact strategies. For example, a humanoid
A. Modular v.s. End-to-End may step onto low obstacles with a single leg, use a two-
leg jump for medium heights, and leverage both arms and
Classical locomotion systems [2], [4] use modular designs
legs for higher obstacles. It remains unclear whether our
(perception, planning, and control) to make the problem
method, without additional references or priors such as [86],
tractable, and can generalize well when their underlying
can automatically discover such diverse contact patterns to
assumptions hold (e.g., low uncertainty, stable foot contacts,
handle a broader range of terrains in higher-DoF systems.
accurate mapping and dynamics modeling). Recent works in-
steadreducesystemcomplexitythroughend-to-endsensor-to-
motorpolicylearning[38],[48],buthavenotyetdemonstrated D. Towards Higher Success Rates on Unseen Terrains
strong generalization to unseen terrains. Westillobservemorefailuresonunseentestterrainsthanon
Our work tries to bridge these two. We use a mapping training terrains, as expected for learning-based systems. No-
moduleandacontrollermodule,andtraintheneuralcontroller tably,mostfailuresoccurduringskilltransitions.Forexample,
end-to-end to generate joint actions from observations. In onTest2therobotmustfirstdecelerateonasparseregionand
the controller, our AME-2 encoder also learns planning-level then execute a climbing maneuver while maintaining precise
representations (global context and local contact-relevant fea- hind-leg footholds. While these specific cases could likely be
tures)forthesubsequentdecoder,playingasimilarroletothe addressedbyfinetuning[48],itremainsunclearwhetherthere
planning module in classical solutions. This design preserves isascalable,principledapproachforlearningsuchchallenging
amodularstructureatthesystemlevelwhileavoidingexplicit skill transitions with zero-shot generalization.
onlinemodel-basedplanning.Inthemappingmodule,wekeep
a simple heuristic structure, but also run a lightweight neural
E. Limitations and Future Work
network in the loop to improve robustness and generalization
under noise, partial observability, and unseen terrain patterns. This work has several limitations. First, we use a 2.5-D
elevationmapanddonotaddressfully3-Dlocomotion.Future
workcouldincorporatemulti-layerelevationmaps[24],which
B. Goal Reaching v.s. Velocity Tracking
aredirectlycompatiblewithourpolicyarchitecture,orexplore
Weuseagoal-reachingformulation[20]insteadofvelocity attention-based voxel representations. Second, our controllers
tracking [15]. This encourages more agile behaviors and arenotdesignedforseverelydegradedperception,suchashigh
reduces reliance on high-frequency navigation commands. A grass or snow. A promising direction is to combine robust

| JOURNALSUBMISSION |     |              |     |     |            |                     |     |             |     |     |            |     |      |             | 17  |
| ----------------- | --- | ------------ | --- | --- | ---------- | ------------------- | --- | ----------- | --- | --- | ---------- | --- | ---- | ----------- | --- |
|                   |     | Ground Truth |     |     |            | Temporal Recurrent  |     |             |     |     |            |     | Ours |             |     |
|                   |     |              |     |     | Estimation |                     |     | Uncertainty |     |     | Estimation |     |      | Uncertainty |     |
Obstacles
Climbing
Consecutive
Stones
Test 1
Test 2
Test 3
Test 4
Fig. 19. Qualitative comparison between the temporal recurrent model and our mapping pipeline on training and test terrains. The visualized maps are
egocentric, with the robot facing rightward. Our mapping pipeline produces accurate estimations when confident and assigns high uncertainty to occluded
regions.Incontrast,thetemporalrecurrentmodelgenerateslessmeaningfuluncertaintymaps,losesfine-graineddetails,andperformsworseonunseenterrains.
controllers [14] with our agile and generalized controllers from ±0 m to ±0.2 m for ANYmal-D, and from ±0 m
into a single scene-aware policy. Third, our mapping module to ±0.15 m for TRON1. Goals are sampled anywhere on
| can fail | in highly | dynamic | environments |     | with | occlusions, | and |     | the terrain. |     |     |     |     |     |     |
| -------- | --------- | ------- | ------------ | --- | ---- | ----------- | --- | --- | ------------ | --- | --- | --- | --- | --- | --- |
could be extended to explicitly reason about moving elements StairDown(Dense,5%):Floatingstairsdescendingwith
•
| in the scene. |     |     |     |     |     |     |     |     | randomwidthsandrandomwallsonbothsides.Theslope |       |             |          |          |            |          |
| ------------- | --- | --- | --- | --- | --- | --- | --- | --- | ---------------------------------------------- | ----- | ----------- | -------- | -------- | ---------- | -------- |
|               |     |     |     |     |     |     |     |     | is increased                                   |       | from 5◦     | to 45◦   | over the | curriculum | for both |
|               |     |     |     |     |     |     |     |     | robots.                                        | Goals | are sampled | anywhere |          | on the     | terrain. |
ACKNOWLEDGMENTS
|     |     |     |     |     |     |     |     | •   | Stair | Up (Dense, | 5%): | Floating |     | stairs ascending | with |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ----- | ---------- | ---- | -------- | --- | ---------------- | ---- |
ThisworkisfundedbyETHAICenter,NCCRautomation,
randomwidthsandrandomwallsonbothsides.Theslope
| Apple Inc. | Any | views, | opinions, | findings, | and | conclusions | or  |     |     |     |     |     |     |     |     |
| ---------- | --- | ------ | --------- | --------- | --- | ----------- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|            |     |        |           |           |     |             |     |     |     |     | 5◦  | 45◦ |     |     |     |
recommendations expressed in this material are those of the is increased from to over the curriculum for both
|                                                         |            |     |                |     |               |     |            |     | robots. | Goals   | are sampled | anywhere     |     | on the    | terrain.  |
| ------------------------------------------------------- | ---------- | --- | -------------- | --- | ------------- | --- | ---------- | --- | ------- | ------- | ----------- | ------------ | --- | --------- | --------- |
| authors                                                 | and should | not | be interpreted |     | as reflecting |     | the views, |     |         |         |             |              |     |           |           |
|                                                         |            |     |                |     |               |     |            |     | Boxes   | (Dense, | 5%):        | Heightfields |     | generated | by adding |
| policiesorposition,eitherexpressedorimplied,ofAppleInc. |            |     |                |     |               |     |            | •   |         |         |             |              |     |           |           |
We thank the help and insights from Yuni Fuchioka, Elena and removing random boxes on flat ground. Over the
|           |         |         |     |        |              |     |          |     | curriculum, |     | the maximum |     | box height | increases | from |
| --------- | ------- | ------- | --- | ------ | ------------ | --- | -------- | --- | ----------- | --- | ----------- | --- | ---------- | --------- | ---- |
| Krasnova, | William | Talbot, |     | Junzhe | He, Matthias |     | Heyrman, |     |             |     |             |     |            |           |      |
ClemensSchwarke,JonasFrey,ManthanPatel,EmilioPalma, 0.05 m to 0.4 m for ANYmal-D, and from 0.05 m to
|         |         |          |          |        |     |        |           |     | 0.3 m | for TRON1. | Goals | are | sampled | anywhere | on the |
| ------- | ------- | -------- | -------- | ------ | --- | ------ | --------- | --- | ----- | ---------- | ----- | --- | ------- | -------- | ------ |
| Vladlen | Koltun, | Matthias | Mu¨ller, | Siyuan |     | Hu. We | thank the |     |       |            |       |     |         |          |        |
terrain.
| support | from Martin | Vechev. |     | We thank | the | technical | support |     |     |     |     |     |     |     |     |
| ------- | ----------- | ------- | --- | -------- | --- | --------- | ------- | --- | --- | --- | --- | --- | --- | --- | --- |
from LimX Dynamics and ANYbotics. • Obstacles (Dense, 5%): Random positive and negative
|     |     |     |           |     |     |     |     |     | obstacles                           | placed | on random |         | slopes.  | Over the | curriculum,  |
| --- | --- | --- | --------- | --- | --- | --- | --- | --- | ----------------------------------- | ------ | --------- | ------- | -------- | -------- | ------------ |
|     |     |     |           |     |     |     |     |     | theobstacledensityincreasesfrom0m−2 |        |           |         |          | to0.5m−2 |              |
|     |     |     | APPENDIXA |     |     |     |     |     |                                     |        |           |         |          |          | for          |
|     |     |     |           |     |     |     |     |     | both robots.                        |        | Goals are | sampled | anywhere | on       | the terrain. |
TERRAINPARAMETERS
|     |     |     |     |     |     |     |     |     | Climbing | Up  | (Climbing, | 20%): | A   | pit to climb | out of, |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | -------- | --- | ---------- | ----- | --- | ------------ | ------- |
•
Weusethefollowingterrainproportionsandparametersfor
|     |     |     |     |     |     |     |     |     | with vertices |     | near the | edges | randomly | perturbed | by up |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------------- | --- | -------- | ----- | -------- | --------- | ----- |
training:
|     |     |     |     |     |     |     |     |     | to 20% | of the | height. | Over | the curriculum, | the | pit height |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | ------ | ------ | ------- | ---- | --------------- | --- | ---------- |
• Rough (Dense, 5%): Heightfields generated from uni- increases from 0.1 m to 1.0 m for ANYmal-D, and from
| form | noise, | with | terrain | resolution | smaller | than | the map |     |       |         |       |        |       |             |        |
| ---- | ------ | ---- | ------- | ---------- | ------- | ---- | ------- | --- | ----- | ------- | ----- | ------ | ----- | ----------- | ------ |
|      |        |      |         |            |         |      |         |     | 0.1 m | to 0.48 | m for | TRON1. | Goals | are sampled | at the |
resolution.Overthecurriculum,thenoiserangeincreases

JOURNALSUBMISSION 18
other end of the terrain. projected gravity, 0.01 rad for joint positions, 1.5 rad/s for
• Climbing Down (Climbing, 5%): A platform to climb joint velocities, and 0.05 m for map observations. We also
down from, with vertices near the edges randomly per- addarandomdriftwithin[−0.03,0.03]mwhenqueryingmap
turbed by up to 20% of the height. Over the curriculum, observations.
the platform height increases from 0.2 m to 1.0 m for Duringstudentpolicytraining,wemake15%ofdepthcloud
ANYmal-D,andfrom0.2mto0.88mforTRON1.Goals points missing and 2% artifacts. We give 10% of the environ-
are sampled at the other end of the terrain. ments access to complete maps with variance 0.0025 m2. We
• Climbing Consecutive (Climbing, 5%): Two stacked corrupt1%pointsinthestudentmapobservationsbyassigning
layers of rings to climb out of. The robot needs to random values with a random variance larger than 1 m2.
consecutivelyclimbuptwiceandthenclimbdowntwice.
Over the curriculum, the first ring height increases from APPENDIXC
0.05 m to 0.5 m for ANYmal-D and from 0.05 m to PPOPARAMETERS
0.3mforTRON1.Thesecondringheightincreasesfrom
We use the following PPO parameters in Table VI.
0.05mto0.4mforANYmal-Dandfrom0.05mto0.3m
for TRON1. Goals are sampled at the other end of the
TABLEVI
terrain.
PPOPARAMETERS
• Gap (Sparse, 5%): A gap to jump over, with vertices
near the edges randomly perturbed by up to 10% of the Parameter Value
ValueLossCoefficient 1.0
distance, and a height difference between the two sides ValueLossClip 0.2
randomlysampledwithin±30%ofthedistance.Overthe EntropyCoefficient Decayfrom0.004to0.001
LearningEpochsperIteration 4
curriculum, the distance increases from 0.1 m to 1.1 m
MiniBatchesperIteration 3
for ANYmal-D and from 0.1 m to 0.6 m for TRON1. SimulationStepsperIteration 24
NumberofEnvironments 4800
Goals are sampled at the other end of the terrain.
LearningRate Adaptive
• Pallets(Sparse,5%):Parallelbeamswithrandomorien- γ 0.99
λ 0.95
tations. Over the curriculum, the beam width decreases
DesiredKLDivergence 0.01
from 0.4 m to 0.16 m for both robots. The gap width Studentonly
LearningRateWhenSurrogateLossDisabled 0.001
increases from 0.08 m to 0.35 m for ANYmal-D and
ActionDistillationLossCoefficient 0.02
from0.08mto0.2mforTRON1.Theinter-beamheight RepresentationLossCoefficient 0.2
difference increases from 0 m to 0.3 m for ANYmal-D
and from 0 m to 0.2 m for TRON1. Goals are sampled
at the other end of the terrain. APPENDIXD
• Stones (Sparse, 30%): We follow the “Stones- TEMPORALRECURRENTBASELINEMAPPINGMODEL
Everywhere” design in [26] and [15]. Goals are sampled
Thearchitectureoftherecurrentmodelusedasthemapping
anywhere on the terrain.
baseline is shown in Fig. 20. We provide the model with
• Beam (Sparse, 5%): A beam connecting two platforms, ground-truth delta transforms, which are sufficient for map
randomly inclined with roll, pitch, and yaw sampled
reconstruction and serve as a clean replacement for the pro-
within ±0.1 rad, and with a platform height difference
prioceptive observations used in prior work [29], [33], where
sampled within ±0.2 m. Over the curriculum, the beam
transforms are not directly available.
width decreases from 0.9 m to 0.18 m for both robots.
Goals are sampled at the other end of the terrain.
Local Grids CNN MLP Estimation Output
For sparse terrains, we add physical floors to half of
them and virtual floors (visible in the map without physical
Concat MLP SRU Uncertainty Output
collision) to the other half. The floor height is randomized
between −1.5 m and −0.35 m. This discourages the robot Delta Transforms MLP
from walking on the floor when traversing shallow sparse
terrains. Fig. 20. The temporal recurrent model architecture to predict egocentric
elevationmapping.
APPENDIXB
RANDOMIZATIONPARAMETERS
REFERENCES
For both robots, we randomize the payload within
[1] P. Fankhauser, M. Bjelonic, C. D. Bellicoso, T. Miki, and M. Hutter,
[−5,5] kg, actuation delays within [0,0.02] s, and friction “Robust rough-terrain locomotion with a quadrupedal robot,” in 2018
coefficients within [0.3,1.0]. For TRON1, we additionally IEEE International Conference on Robotics and Automation (ICRA).
IEEE,2018,pp.5761–5768.
apply random biases to PD gains and armatures within ±15%
[2] F.Jenelten,R.Grandia,F.Farshidian,andM.Hutter,“Tamols:Terrain-
of the raw values. aware motion optimization for legged systems,” IEEE Transactions on
For policy observations, we add zero-mean uniform noise Robotics,vol.38,no.6,pp.3395–3413,2022.
[3] S. Fahmi, V. Barasuol, D. Esteban, O. Villarreal, and C. Semini,
with the following maximum magnitudes: 0.1 m/s for base
“Vital:Vision-basedterrain-awarelocomotionforleggedrobots,”IEEE
linear velocity, 0.2 rad/s for base angular velocity, 0.05 for TransactionsonRobotics,vol.39,no.2,pp.885–904,2022.

JOURNALSUBMISSION 19
[4] R. Grandia, F. Jenelten, S. Yang, F. Farshidian, and M. Hutter, “Per- [26] C.Zhang,N.Rudin,D.Hoeller,andM.Hutter,“Learningagilelocomo-
ceptive locomotion through nonlinear model-predictive control,” IEEE tion on risky terrains,” in 2024 IEEE/RSJ International Conference on
TransactionsonRobotics,vol.39,no.5,pp.3402–3421,2023. IntelligentRobotsandSystems(IROS). IEEE,2024,pp.11864–11871.
[5] R. Akizhanov, V. Dhe´din, M. Khadiv, and I. Laptev, “Learning [27] F. Shi, C. Zhang, T. Miki, J. Lee, M. Hutter, and S. Coros, “Re-
feasible transitions for efficient contact planning,” arXiv preprint thinking robustness assessment: Adversarial attacks on learning-based
arXiv:2407.11788,2024. quadrupedallocomotioncontrollers,”inRobotics:ScienceandSystems
[6] R. J. Griffin, G. Wiedebach, S. McCrory, S. Bertrand, I. Lee, and XX, Delft, The Netherlands, July 15-19, 2024, D. Kulic, G. Venture,
J.Pratt,“Footstepplanningforautonomouswalkingoverroughterrain,” K.E.Bekris,andE.Coronado,Eds.,2024.
in2019IEEE-RAS19thInternationalConferenceonHumanoidRobots [28] J.Lee,M.Bjelonic,A.Reske,L.Wellhausen,T.Miki,andM.Hutter,
(Humanoids),2019,pp.9–16. “Learning robust autonomous navigation and locomotion for wheeled-
[7] C. Mastalli, W. Merkt, G. Xin, J. Shim, M. Mistry, I. Havoutis, and leggedrobots,”ScienceRobotics,vol.9,no.89,p.eadi9641,2024.
S.Vijayakumar,“Agilemaneuversinleggedrobots:apredictivecontrol [29] J.Sun,G.Han,P.Sun,W.Zhao,J.Cao,J.Wang,Y.Guo,andQ.Zhang,
approach,”arXivpreprintarXiv:2203.07554,2022. “Dpl: Depth-only perceptive humanoid locomotion via realistic depth
[8] F. Jenelten, T. Miki, A. E. Vijayan, M. Bjelonic, and M. Hutter, synthesis and cross-attention terrain reconstruction,” arXiv preprint
“Perceptivelocomotioninroughterrain–onlinefootholdoptimization,” arXiv:2510.07152,2025.
IEEE Robotics and Automation Letters, vol. 5, no. 4, pp. 5370–5376, [30] Y. Yang, G. Shi, C. Lin, X. Meng, R. Scalise, M. G. Castro, W. Yu,
2020. T.Zhang,D.Zhao,J.Tanetal.,“Agilecontinuousjumpingindiscon-
[9] F. Jenelten, J. Hwangbo, F. Tresoldi, C. D. Bellicoso, and M. Hutter, tinuous terrains,” in 2025 IEEE International Conference on Robotics
“Dynamiclocomotiononslipperyground,”IEEERoboticsandAutoma- andAutomation(ICRA). IEEE,2025,pp.10245–10252.
tionLetters,vol.4,no.4,pp.4170–4176,2019. [31] D. Hoeller, N. Rudin, C. Choy, A. Anandkumar, and M. Hutter,
[10] P.Fankhauser,M.Bloesch,andM.Hutter,“Probabilisticterrainmapping “Neuralscenerepresentationforlocomotiononstructuredterrain,”IEEE
for mobile robots with uncertain localization,” IEEE Robotics and RoboticsandAutomationLetters,vol.7,no.4,pp.8667–8674,2022.
AutomationLetters,vol.3,no.4,pp.3019–3026,2018. [32] D. Hoeller, N. Rudin, D. Sako, and M. Hutter, “Anymal parkour:
[11] T. Miki, L. Wellhausen, R. Grandia, F. Jenelten, T. Homberger, and Learning agile navigation for quadrupedal robots,” Science Robotics,
M. Hutter, “Elevation mapping for locomotion and navigation using vol.9,no.88,p.eadi7566,2024.
gpu,”in2022IEEE/RSJInternationalConferenceonIntelligentRobots [33] R. Yu, Q. Wang, Y. Wang, Z. Wang, J. Wu, and Q. Zhu, “Walking
andSystems(IROS),2022,pp.2273–2280. withterrainreconstruction:Learningtotraverseriskysparsefootholds,”
[12] S. Ha, J. Lee, M. van de Panne, Z. Xie, W. Yu, and M. Khadiv, arXivpreprintarXiv:2409.15692,2024.
“Learning-basedleggedlocomotion:Stateoftheartandfutureperspec- [34] H. Duan, B. Pandit, M. S. Gadde, B. Van Marum, J. Dao, C. Kim,
tives,” The International Journal of Robotics Research, vol. 44, no. 8, and A. Fern, “Learning vision-based bipedal locomotion for challeng-
pp.1396–1427,2025. ing terrain,” in 2024 IEEE International Conference on Robotics and
[13] N. Rudin, D. Hoeller, P. Reist, and M. Hutter, “Learning to walk Automation(ICRA). IEEE,2024,pp.56–62.
in minutes using massively parallel deep reinforcement learning,” in [35] A. Agarwal, A. Kumar, J. Malik, and D. Pathak, “Legged locomotion
Conferenceonrobotlearning. PMLR,2022,pp.91–100. inchallengingterrainsusingegocentricvision,”inProceedingsofThe
[14] T.Miki,J.Lee,J.Hwangbo,L.Wellhausen,V.Koltun,andM.Hutter, 6thConferenceonRobotLearning. PMLR,2023.
“Learning robust perceptive locomotion for quadrupedal robots in the [36] X.Cheng,K.Shi,A.Agarwal,andD.Pathak,“Extremeparkourwith
wild,”Sciencerobotics,vol.7,no.62,p.eabk2822,2022. leggedrobots,”in2024IEEEInternationalConferenceonRoboticsand
[15] J. He, C. Zhang, F. Jenelten, R. Grandia, M. Ba¨cher, and M. Hutter, Automation(ICRA). IEEE,2024,pp.11443–11450.
“Attention-basedmapencodingforlearninggeneralizedleggedlocomo- [37] R.Yang,M.Zhang,N.Hansen,H.Xu,andX.Wang,“Learningvision-
tion,”ScienceRobotics,vol.10,no.105,p.eadv3604,2025. guidedquadrupedallocomotionend-to-endwithcross-modaltransform-
[16] A. Allshire, H. Choi, J. Zhang, D. McAllister, A. Zhang, C. M. Kim, ers,”inInternationalConferenceonLearningRepresentations,2022.
T. Darrell, P. Abbeel, J. Malik, and A. Kanazawa, “Visual imitation [38] Z. Zhuang, S. Yao, and H. Zhao, “Humanoid parkour learning,” in
enablescontextualhumanoidcontrol,”arXivpreprintarXiv:2505.03729, ConferenceonRobotLearning. PMLR,2025,pp.1975–1991.
2025. [39] S. Li, S. Luo, J. Wu, and Q. Zhu, “Move: Multi-skill omnidirectional
[17] J.Long,J.Ren,M.Shi,Z.Wang,T.Huang,P.Luo,andJ.Pang,“Learn- leggedlocomotionwithlimitedviewin3denvironments,”in2025IEEE
inghumanoidlocomotionwithperceptiveinternalmodel,”in2025IEEE InternationalConferenceonRoboticsandAutomation(ICRA),2025,pp.
InternationalConferenceonRoboticsandAutomation(ICRA). IEEE, 7647–7653.
2025,pp.9997–10003. [40] Q. Ben, B. Xu, K. Li, F. Jia, W. Zhang, J. Wang, J. Wang,
[18] J.Ren,T.Huang,H.Wang,Z.Wang,Q.Ben,J.Long,Y.Yang,J.Pang, D. Lin, and J. Pang, “Gallant: Voxel grid-based humanoid locomotion
andP.Luo,“Vb-com:Learningvision-blindcompositehumanoidloco- and local-navigation across 3d constrained terrains,” arXiv preprint
motionagainstdeficientperception,”arXivpreprintarXiv:2502.14814, arXiv:2511.14625,2025.
2025. [41] S.Luo,S.Li,R.Yu,Z.Wang,J.Wu,andQ.Zhu,“Pie:Parkourwith
[19] H.Wang,Z.Wang,J.Ren,Q.Ben,T.Huang,W.Zhang,andJ.Pang, implicit-explicitlearningframeworkforleggedrobots,”IEEERobotics
“Beamdojo:Learningagilehumanoidlocomotiononsparsefootholds,” andAutomationLetters,vol.9,no.11,pp.9986–9993,2024.
inRobotics:ScienceandSystems(RSS),2025. [42] E. Chane-Sane, J. Amigo, T. Flayols, L. Righetti, and N. Mansard,
[20] N.Rudin,D.Hoeller,M.Bjelonic,andM.Hutter,“Advancedskillsby “Soloparkour:Constrainedreinforcementlearningforvisuallocomotion
learninglocomotionandlocalnavigationend-to-end,”in2022IEEE/RSJ from privileged experience,” in Proceedings of The 8th Conference on
International Conference on Intelligent Robots and Systems (IROS), Robot Learning, ser. Proceedings of Machine Learning Research, vol.
2022,pp.2497–2503. 270. PMLR,2025,pp.4268–4285.
[21] F. Jenelten, J. He, F. Farshidian, and M. Hutter, “Dtc: Deep tracking [43] S.Kareer,N.Yokoyama,D.Batra,S.Ha,andJ.Truong,“Vinl:Visual
control,”ScienceRobotics,vol.9,no.86,p.eadh5401,2024. navigationandlocomotionoverobstacles,”in2023IEEEInternational
[22] S.Gangapurwala,M.Geisert,R.Orsolino,M.Fallon,andI.Havoutis, Conference on Robotics and Automation (ICRA). IEEE, 2023, pp.
“Rloc: Terrain-aware legged locomotion using reinforcement learning 2018–2024.
and optimal control,” IEEE Transactions on Robotics, vol. 38, no. 5, [44] H. Lai, J. Cao, J. Xu, H. Wu, Y. Lin, T. Kong, Y. Yu, and W. Zhang,
pp.2908–2927,2022. “Worldmodel-basedperceptionforvisualleggedlocomotion,”in2025
[23] Y.Dong,J.Ma,L.Zhao,W.Li,andP.Lu,“Marg:Masteringriskygap IEEE International Conference on Robotics and Automation (ICRA),
terrains for legged robots with elevation mapping,” IEEE Transactions 2025,pp.11531–11537.
onRobotics,2025. [45] P. Li, H. Li, Y. Ma, L. Chang, X. Yang, R. Yu, Y. Zhang, Y. Cao,
[24] Y.Chen,J.Ma,Z.Luo,Y.Han,Y.Dong,B.Xu,andP.Lu,“Learning Q. Zhu, and G. Sartoretti, “Kivi: Kinesthetic-visuospatial integration
autonomous and safe quadruped traversal of complex terrains using for dynamic and safe egocentric legged locomotion,” arXiv preprint
multi-layer elevation maps,” IEEE Robotics and Automation Letters, arXiv:2509.23650,2025.
2025. [46] H.Su,H.Luo,S.Yang,K.Jiang,W.Zhang,andH.Chen,“Lipm-guided
[25] T. Miki, J. Lee, L. Wellhausen, and M. Hutter, “Learning to walk in reinforcementlearningforstableandperceptivelocomotioninbipedal
confined spaces using 3d representation,” in 2024 IEEE International robots,”in2025IEEE-RAS24thInternationalConferenceonHumanoid
ConferenceonRoboticsandAutomation(ICRA),2024,pp.8649–8656. Robots(Humanoids),2025,pp.1031–1038.

| JOURNALSUBMISSION |     |     |     |     |     |     |     |     |     |     |     |     |     | 20  |
| ----------------- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
[47] D.Wang,X.Wang,X.Liu,J.Shi,Y.Zhao,C.Bai,andX.Li,“More: T. Howard, and N. Atanasov, Eds., 2018. [Online]. Available:
Mixture of residual experts for humanoid lifelike gaits learning on http://www.roboticsproceedings.org/rss14/p08.html
complexterrains,”arXivpreprintarXiv:2506.08840,2025. [68] Z.Chen,M.Ji,X.Cheng,X.Peng,X.B.Peng,andX.Wang,“Gmt:
[48] N.Rudin,J.He,J.Aurand,andM.Hutter,“Parkourinthewild:Learning General motion tracking for humanoid whole-body control,” arXiv
a general and extensible agile locomotion policy using multi-expert preprintarXiv:2506.14770,2025.
distillationandrlfine-tuning,”arXivpreprintarXiv:2505.11164,2025. [69] N. Messikommer, J. Xing, E. Aljalbout, and D. Scaramuzza,
|     |     |     |     |     |     |     |     | “Student-informed |     | teacher | training,” | in 7th | Robot Learning | Workshop: |
| --- | --- | --- | --- | --- | --- | --- | --- | ----------------- | --- | ------- | ---------- | ------ | -------------- | --------- |
[49] T.He,C.Zhang,W.Xiao,G.He,C.Liu,andG.Shi,“Agilebutsafe:
Learning collision-free high-speed legged locomotion,” in Proceedings TowardsRobotswithHuman-LevelAbilities,2025.[Online].Available:
ofRobotics:ScienceandSystems,Delft,Netherlands,July2024. https://openreview.net/forum?id=q06EAFOwDI
[50] C. Zhang, J. Jin, J. Frey, N. Rudin, M. Mattamala, C. Cadena, and [70] J.Lee,J.Hwangbo,L.Wellhausen,V.Koltun,andM.Hutter,“Learning
M. Hutter, “Resilient legged local navigation: Learning to traverse quadrupedal locomotion over challenging terrain,” Science robotics,
vol.5,no.47,p.eabc5986,2020.
withcompromisedperceptionend-to-end,”in2024IEEEInternational
ConferenceonRoboticsandAutomation(ICRA),2024,pp.34–41. [71] T.Pfau,A.G.deRivaz,S.Brighton,andR.Weller,“Kineticsofjump
[51] A. Kendall and Y. Gal, “What uncertainties do we need in bayesian landing in agility dogs,” The Veterinary Journal, vol. 190, no. 2, pp.
| deep | learning | for computer | vision?” |     | Advances | in neural | information | 278–283,2011. |     |     |     |     |     |     |
| ---- | -------- | ------------ | -------- | --- | -------- | --------- | ----------- | ------------- | --- | --- | --- | --- | --- | --- |
processingsystems,vol.30,2017. [72] S. McErlain-Naylor, M. King, and S. Allen, “Surface acceleration
transmissionduringdroplandingsinhumans,”JournalofBiomechanics,
[52] S.Jung,D.Gwak,B.Boots,andJ.Hays,“Uncertainty-awareaccurate
vol.118,p.110269,2021.
| elevation | modeling | for | off-road | navigation | via | neural processes,” |     | in  |     |     |     |     |     |     |
| --------- | -------- | --- | -------- | ---------- | --- | ------------------ | --- | --- | --- | --- | --- | --- | --- | --- |
ConferenceonRobotLearning. PMLR,2025,pp.2802–2822. [73] J. Tobin, R. Fong, A. Ray, J. Schneider, W. Zaremba, and P. Abbeel,
[53] H.Wang,Z.Shi,C.Zhu,Y.Qiao,C.Zhang,F.Yang,P.Ren,L.Lu,and “Domain randomization for transferring deep neural networks from
simulationtotherealworld,”in2017IEEE/RSJinternationalconference
| D. Xuan,   | “Integrating |            | learning-based |     | manipulation    | and physics-based |          |                                                                  |     |     |     |     |                     |     |
| ---------- | ------------ | ---------- | -------------- | --- | --------------- | ----------------- | -------- | ---------------------------------------------------------------- | --- | --- | --- | --- | ------------------- | --- |
|            |              |            |                |     |                 |                   |          | onintelligentrobotsandsystems(IROS).                             |     |     |     |     | IEEE,2017,pp.23–30. |     |
| locomotion | for          | whole-body | badminton      |     | robot control,” | arXiv             | preprint |                                                                  |     |     |     |     |                     |     |
|            |              |            |                |     |                 |                   |          | [74] C.Schwarke,M.Mittal,N.Rudin,D.Hoeller,andM.Hutter,“Rsl-rl:A |     |     |     |     |                     |     |
arXiv:2504.17771,2025.
learninglibraryforroboticsresearch,”arXivpreprintarXiv:2509.10771,
[54] M.Hutter,C.Gehring,D.Jud,A.Lauber,C.D.Bellicoso,V.Tsounis,
| J. Hwangbo,                                               |        | K. Bodie,   | P. Fankhauser, |     | M. Bloesch | et al.,          | “Anymal-a | 2025.                                                              |                   |     |           |         |       |                   |
| --------------------------------------------------------- | ------ | ----------- | -------------- | --- | ---------- | ---------------- | --------- | ------------------------------------------------------------------ | ----------------- | --- | --------- | ------- | ----- | ----------------- |
|                                                           |        |             |                |     |            |                  |           | [75] O.R.developers,“Onnxruntime,”https://onnxruntime.ai/,2025.    |                   |     |           |         |       |                   |
| highly                                                    | mobile | and dynamic | quadrupedal    |     | robot,”    | in 2016 IEEE/RSJ | in-       |                                                                    |                   |     |           |         |       |                   |
|                                                           |        |             |                |     |            |                  |           | [76] J.Hwangbo,J.Lee,A.Dosovitskiy,D.Bellicoso,V.Tsounis,V.Koltun, |                   |     |           |         |       |                   |
| ternationalconferenceonintelligentrobotsandsystems(IROS). |        |             |                |     |            |                  | IEEE,     |                                                                    |                   |     |           |         |       |                   |
|                                                           |        |             |                |     |            |                  |           | and M.                                                             | Hutter, “Learning |     | agile and | dynamic | motor | skills for legged |
2016,pp.38–44.
robots,”ScienceRobotics,vol.4,no.26,p.eaau5872,2019.
[55] LimXDynamics,“TRON1:Thefirstmulti-modalbipedrobot,”https:
//www.limxdynamics.com/en/tron1,2025,accessed:2025-11-23. [77] Y.-H. Shin, T.-G. Song, G. Ji, and H.-W. Park, “Actuator-constrained
|            |         |       |        |                   |     |             |          | reinforcement | learning | for | high-speed | quadrupedal | locomotion,” | arXiv |
| ---------- | ------- | ----- | ------ | ----------------- | --- | ----------- | -------- | ------------- | -------- | --- | ---------- | ----------- | ------------ | ----- |
| [56] H. J. | Lee, S. | Hong, | and S. | Kim, “Integrating |     | model-based | footstep |               |          |     |            |             |              |       |
preprintarXiv:2312.17507,2023.
| planning | with | model-free | reinforcement |     | learning | for dynamic | legged |                                                                     |     |     |     |     |     |     |
| -------- | ---- | ---------- | ------------- | --- | -------- | ----------- | ------ | ------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|          |      |            |               |     |          |             |        | [78] O.Ronneberger,P.Fischer,andT.Brox,“U-net:Convolutionalnetworks |     |     |     |     |     |     |
locomotion,”in2024IEEE/RSJInternationalConferenceonIntelligent
|                         |     |     |     |                           |     |     |     | for biomedical                                         | image | segmentation,” |     | in International |     | Conference on |
| ----------------------- | --- | --- | --- | ------------------------- | --- | --- | --- | ------------------------------------------------------ | ----- | -------------- | --- | ---------------- | --- | ------------- |
| RobotsandSystems(IROS). |     |     |     | IEEE,2024,pp.11248–11255. |     |     |     |                                                        |       |                |     |                  |     |               |
|                         |     |     |     |                           |     |     |     | Medicalimagecomputingandcomputer-assistedintervention. |       |                |     |                  |     | Springer,     |
[57] W.Yu,D.Jain,A.Escontrela,A.Iscen,P.Xu,E.Coumans,S.Ha,J.Tan, 2015,pp.234–241.
andT.Zhang,“Visual-locomotion:Learningtowalkoncomplexterrains
|     |     |     |     |     |     |     |     | [79] M. Macklin, | “Warp: | A high-performance |     |     | python framework | for gpu |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------------- | ------ | ------------------ | --- | --- | ---------------- | ------- |
withvision,”in5thAnnualConferenceonRobotLearning,2021.
|     |     |     |     |     |     |     |     | simulation | and graphics,” |     | https://github.com/nvidia/warp, |     |     | March 2022, |
| --- | --- | --- | --- | --- | --- | --- | --- | ---------- | -------------- | --- | ------------------------------- | --- | --- | ----------- |
[58] R.Yang,G.Yang,andX.Wang,“Neuralvolumetricmemoryforvisual
nVIDIAGPUTechnologyConference(GTC).
| locomotion | control,” |     | in Proceedings | of  | the IEEE/CVF | conference |     | on                                                                  |     |     |     |     |     |     |
| ---------- | --------- | --- | -------------- | --- | ------------ | ---------- | --- | ------------------------------------------------------------------- | --- | --- | --- | --- | --- | --- |
|            |           |     |                |     |              |            |     | [80] M.Seitzer,A.Tavakoli,D.Antic,andG.Martius,“Onthepitfallsofhet- |     |     |     |     |     |     |
computervisionandpatternrecognition,2023,pp.1430–1440. eroscedasticuncertaintyestimationwithprobabilisticneuralnetworks,”
[59] H.Kim,A.Mnih,J.Schwarz,M.Garnelo,A.Eslami,D.Rosenbaum,
inInternationalConferenceonLearningRepresentations,2022.
O.Vinyals,andY.W.Teh,“Attentiveneuralprocesses,”arXivpreprint
|     |     |     |     |     |     |     |     | [81] A. Chambolle, | “An | algorithm | for | total variation |     | minimization and |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------ | --- | --------- | --- | --------------- | --- | ---------------- |
arXiv:1901.05761,2019.
|     |     |     |     |     |     |     |     | applications,” | Journal | of Mathematical |     | imaging | and | vision, vol. 20, |
| --- | --- | --- | --- | --- | --- | --- | --- | -------------- | ------- | --------------- | --- | ------- | --- | ---------------- |
[60] J.Schulman,F.Wolski,P.Dhariwal,A.Radford,andO.Klimov,“Prox-
no.1,pp.89–97,2004.
imalpolicyoptimizationalgorithms,”arXivpreprintarXiv:1707.06347, [82] S. Khattak, T. Homberger, L. Bernreiter, J. Nubert, O. Andersson,
2017. R. Siegwart, K. Alexis, and M. Hutter, “Compslam: Complementary
[61] V.Makoviychuk,L.Wawrzyniak,Y.Guo,M.Lu,K.Storey,M.Macklin,
|             |     |        |              |     |          |             |           | hierarchical | multi-modal | localization |     | and mapping | for | robot autonomy |
| ----------- | --- | ------ | ------------ | --- | -------- | ----------- | --------- | ------------ | ----------- | ------------ | --- | ----------- | --- | -------------- |
| D. Hoeller, | N.  | Rudin, | A. Allshire, | A.  | Handa et | al., “Isaac | gym: High |              |             |              |     |             |     |                |
inundergroundenvironments,”arXivpreprintarXiv:2505.06483,2025.
| performance |     | gpu-based | physics | simulation | for | robot learning,” | arXiv |                                                                    |     |     |     |     |     |     |
| ----------- | --- | --------- | ------- | ---------- | --- | ---------------- | ----- | ------------------------------------------------------------------ | --- | --- | --- | --- | --- | --- |
|             |     |           |         |            |     |                  |       | [83] J.Nubert,S.Khattak,andM.Hutter,“Graph-basedmulti-sensorfusion |     |     |     |     |     |     |
preprintarXiv:2108.10470,2021. forconsistentlocalizationofautonomousconstructionrobots,”in2022
[62] X. B. Peng and M. van de Panne, “Learning locomotion skills using InternationalConferenceonRoboticsandAutomation(ICRA). IEEE,
deeprl:doesthechoiceofactionspacematter?”inProceedingsofthe
2022,pp.10048–10054.
ACMSIGGRAPH/EurographicsSymposiumonComputerAnimation,
|     |     |     |     |     |     |     |     | [84] K.Chen,R.Nemiroff,andB.T.Lopez,“Directlidar-inertialodometry: |     |     |     |     |     |     |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------------------------------------------------------------ | --- | --- | --- | --- | --- | --- |
ser. SCA ’17. New York, NY, USA: Association for Computing Lightweightliowithcontinuous-timemotioncorrection,”in2023IEEE
Machinery,2017. InternationalConferenceonRoboticsandAutomation(ICRA). IEEE,
[63] A.Vaswani,N.Shazeer,N.Parmar,J.Uszkoreit,L.Jones,A.N.Gomez, 2023,pp.3983–3989.
| Ł. Kaiser, | and | I. Polosukhin, |     | “Attention | is all | you need,” | Advances | in             |     |          |          |           |          |           |
| ---------- | --- | -------------- | --- | ---------- | ------ | ---------- | -------- | -------------- | --- | -------- | -------- | --------- | -------- | --------- |
|            |     |                |     |            |        |            |          | [85] Z. Zhuang | and | H. Zhao, | “Embrace | contacts: | humanoid | shadowing |
neuralinformationprocessingsystems,vol.30,2017.
|                                                              |     |     |     |     |     |     |     | with full      | body ground | contacts,” | in  | 9th Annual | Conference | on Robot |
| ------------------------------------------------------------ | --- | --- | --- | --- | --- | --- | --- | -------------- | ----------- | ---------- | --- | ---------- | ---------- | -------- |
| [64] C.Zhang,W.Xiao,T.He,andG.Shi,“Wococo:Learningwhole-body |     |     |     |     |     |     |     | Learning,2025. |             |            |     |            |            |          |
humanoidcontrolwithsequentialcontacts,”inProceedingsofThe8th [86] L. Yang, X. Huang, Z. Wu, A. Kanazawa, P. Abbeel, C. Sferrazza,
Conference on Robot Learning, ser. Proceedings of Machine Learning C. K. Liu, R. Duan, and G. Shi, “Omniretarget: Interaction-preserving
Research,vol.270. PMLR,2025,pp.455–472. datagenerationforhumanoidwhole-bodyloco-manipulationandscene
| [65] T. He, | Z. Luo, | X. He, | W. Xiao, | C. Zhang, | W.  | Zhang, K. | M. Kitani, |     |     |     |     |     |     |     |
| ----------- | ------- | ------ | -------- | --------- | --- | --------- | ---------- | --- | --- | --- | --- | --- | --- | --- |
interaction,”arXivpreprintarXiv:2509.26633,2025.
C. Liu, and G. Shi, “Omnih2o: Universal and dexterous human-to- [87] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai,
humanoid whole-body teleoperation and learning,” in Proceedings of T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly,
The 8th Conference on Robot Learning, ser. Proceedings of Machine J. Uszkoreit, and N. Houlsby, “An image is worth 16x16 words:
LearningResearch,vol.270. PMLR,November2025,pp.1516–1540. Transformersforimagerecognitionatscale,”ICLR,2021.
[Online].Available:https://proceedings.mlr.press/v270/he25b.html
|     |     |     |     |     |     |     |     | [88] F. Yang, | P. Frivik, | D. Hoeller, | C.  | Wang, | C. Cadena, | and M. Hutter, |
| --- | --- | --- | --- | --- | --- | --- | --- | ------------- | ---------- | ----------- | --- | ----- | ---------- | -------------- |
[66] Z. Li, X. B. Peng, P. Abbeel, S. Levine, G. Berseth, and K. Sreenath, “Spatially-enhanced recurrent memory for long-range mapless naviga-
“Reinforcement learning for versatile, dynamic, and robust bipedal tion via end-to-end reinforcement learning,” The International Journal
locomotion control,” The International Journal of Robotics Research, ofRoboticsResearch,p.02783649251401926,2025.
2024. [89] T.He,W.Xiao,T.Lin,Z.Luo,Z.Xu,Z.Jiang,J.Kautz,C.Liu,G.Shi,
[67] L. Pinto, M. Andrychowicz, P. Welinder, W. Zaremba, and P. Abbeel, X.Wang,L.J.Fan,andY.Zhu,“HOVER:versatileneuralwhole-body
“Asymmetricactorcriticforimage-basedrobotlearning,”inRobotics: controller for humanoid robots,” in IEEE International Conference on
Science and Systems XIV, Carnegie Mellon University, Pittsburgh, RoboticsandAutomation,ICRA. IEEE,2025,pp.9989–9996.
Pennsylvania,USA,June26-30,2018,H.Kress-Gazit,S.S.Srinivasa,