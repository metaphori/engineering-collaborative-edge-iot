# Engineering Resilient Collaborative Edge-enabled IoT

## How-to: basics

- Simulation descriptors are located under `src/main/yaml`
- Simulations are started through commands like:
```commandline
$ ./gradlew --no-daemon && \
  java -Xmx5024m -cp "build/classes/main:build/resources/main" \
  it.unibo.alchemist.Alchemist \
  -b -var <VAR1> <VAR2> ... \
  -y <simulationDescriptor> -e <baseFilepathForDataFiles> \
  -t <simulationDuration> -p <numOfParallelRuns> -v &> exec.txt &
```
- Data can be plotted through command
```commandline
$ ./plotter.py <basedir> <basefilename> <plotGenDescriptor>
```
- See `plot.yml` for an example of plot generation descriptor

## Experiments in the paper
 
### Figure 5

The experiment aims to compare the performance of smart vs. naive coordinators.
A smart coordinator employs a task allocation model which match problems with more performant solvers.

Notes:

- Naive allocation: only distinguish between non-available and available solvers
- Smart allocation: only distinguishes between 'smart' and 'normal' solvers
- The increased performance of 'smart solvers' is hardcoded in method `execute`
- The simulation does not take into account the location of problems and solvers
- This simulation considers also the disconnection of coordinators for 10s
- 50 runs

#### Generate data

```commandline
$ ./gradlew --no-daemon && java -Xmx5024m  \
  -cp "build/classes/main:build/resources/main"  \
  it.unibo.alchemist.Alchemist  \
  -b -var smartness random  \
  -y src/main/yaml/casestudy.yml -e data/20181113f -t 800 -p 3 -v &> exec.txt &
```

#### Generate plots

```commandline
$ ./plotter.py data 20181113f plot.yml
```

#### Put plots into a grid

```commandline
$ montage -tile 2x -geometry +0+0 data/imgs/20181113f*.png montage20181113f.png && xdg-open montage20181113f.png
```