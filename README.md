# Cheetah - the CODAR Experiment Harness

## Overview
* Cheetah is designed to run Exascale science applications
  using different parameters and components to determine the best combination
  for deployment on different supercomputers.
* To use Cheetah, the user first writes a <b>campaign specification file</b> in python.
  In this file, one specifies:
  * which MPI programs to launch in parallel
  * what computing resources to give to those programs
  * which configuration files to copy to each experiment's directory
  * lists of possible values for parameters (including computing resources); the cartesian product of those parameter spaces defines the number of experiments in the campaign 
  * how many times to repeat each experiment to collect enough statistics
    to estimate the variability of the results.
* The campaign specification file does not define how programs communicate with each other.
  This is done in <b>adios2.xml</b> that can be used to glue the input from one program
  with the output from another provided that all the involved programs are doing I/O using [ADIOS2](https://adios2.readthedocs.io/en/latest/index.html).
  The data exchange can happen at each time step as the data is generated by producers.
  ADIOS2 shields a user from I/O details. Without recompiling a program, in <b>adios2.xml</b> one
  can specify that output/input is written/read to/from a file or network socket, whether a producer should wait
  for a certain number of reads by consumers or discard data after some timeout if consumers cannot keep up, etc.
* To generate shell scripts and configuration files for the campaign and the particular supercomputer,
  one executes  
  ```bash
  cheetah.py create-campaign -a <dir with configs & binaries> \
  	     -o <campaign dir> -e <specification>.py -m <supercomputer>
  ```  
* The experiments are organized into separate subdirectories that contain
  all the necessary executables and configuration files to run each experiment separately if necessary.
  Each such directory would also contain its own log files so that one can later examine what went
  wrong with a particular experiment.
* The top level directory of the campaign contains <b>run-all.sh</b> file that one can use to launch
  the whole set of experiments on the allocated resources. Allocation is done by a user outside of Cheetah using
  the resource manager of a particular supercomputer.
* There is also <b>cancel.sh</b> that allows to stop the whole campaign.
* As campaign runs, one can examine its progress with
  ```bash
  cheetah.py status <campaign dir>
  ```
  or generate a detailed report with
  ```bash
  cheetah.py generate-report <campaign dir>
  ```
  Use `-h` option for a particular command to learn more details

## Cheetah installation
* Dependency: Linux, python 3.5+
* On supercomputers it should be installed on a parallel file system visible from compute/mother nodes
* One can also run campaign on a standalone computer by using <b>local</b> as a supercomputer
* To install Cheetah, do:
  ```bash
  git clone git@github.com:CODARcode/cheetah.git
  cd cheetah          
  python3 -m venv venv-cheetah
  source venv-cheetah/bin/activate
  pip install --editable .
  ```
* Cheetah was tested so far on <b>Summit</b>, <b>Theta</b>, standalone Linux computers
* It can also be installed with [spack](https://spack.io):
  ```bash
  spack install codar-cheetah
  ```
  (so far this does not work for me??)

## Setting up Cheetah environment
   ```bash
   source <cheetah dir>/venv-cheetah/bin/activate
   ```

## Campaign specification file
* Here is a small example of a campaign file:
  ```python
  from codar.cheetah import Campaign
  from codar.cheetah import parameters as p
  from datetime import timedelta

  class GrayScott(Campaign):
    name = "Gray-Scott-A"
    codes = [("gray-scott", dict(exe="gray-scott", sleep_after=1)),
             ("compression", dict(exe="compression")) ]
    supported_machines = ['local', 'theta']
    scheduler_options = {
        "theta": {
            "queue": "debug-flat-quad",
            "project": "XXXX",
        }
    }
    umask = '027'
    sweeps = [
     p.SweepGroup(name="Gray-Scott",
                  walltime=timedelta(minutes=30),
                  component_subdirs=True,
                  run_repetitions=2,
                  component_inputs={
                      'gray-scott': ['settings.json','adios2.xml'],
                      'compression': ['adios2.xml','sz.config','zc.config']
                  },
      parameter_groups=
      [p.Sweep([
          p.ParamCmdLineArg("gray-scott", "settings", 1, ["settings.json"]),
          p.ParamConfig("gray-scott", "L", "settings.json", "L",
                          [32, 64]),
          p.ParamConfig("gray-scott", "noise", "settings.json", "noise",
                          [0.01, 0.1]),
          p.ParamRunner('gray-scott', 'nprocs', [4] ),
          p.ParamCmdLineArg("compression", "input", 1, ["../gray-scott/gs.bp"]),
          p.ParamCmdLineArg("compression", "output", 2, ["compression.bp"]),
          p.ParamRunner('compression', 'nprocs', [1] )          
        ]),
      ]),
    ]
  ```
* One needs to import <b>Campaign</b> and <b>parameters</b> modules.
* To create your own campaign, you need to inherit from <b>Campaign</b> class and overwrite some of its fields:
  - <b>name</b> - campaign name
  - <b>codes</b> - what MPI programs to run in parallel.
    It is a dictionary mapping the campaign name of a program to the corresponding binary,
    possibly setting up some other parameters. In this example, `sleep_after=1` means that
    <b>gray-scott</b> started 1 second earlier than <b>compression</b> (is that right?? or the other way around??)
  - <b>supported_machines</b> - indicates for which supercomputer this campaign can be generated (why is it needed considering
    that it is defined when campaign is generated from the specification file??)
  - <b>scheduler_options</b> - defines some extra options to the resource manager not defined in <b>Savanna</b> such as project to charge the run to;
    note: <b>Savanna</b> is part of Cheetah that shields a user from the pecularities of a supercomputer
  - <b>umask</b> specifies the permissions for the newly created campaign files and directories.
  - <b>sweeps</b> is a list of <b>SweepGroups</b>
    + <b>SweepGroup</b> has a  name, a list of configuration files to copy into each experiment's directory,
      <b>parameter_groups</b>.
      * <b>parameter_groups</b> is a list of `Sweeps` where one specifies with which parameters to run experiments.
      * Some parameters are fixed values, and some are lists. A cartesian product of all the parameters is taken
	to compute the experiments to perform.
  - Examples of parameter types:
    + <b>ParamCmdLineArg</b> allows to specify command line positional parameter for a particular program.
      For example
      ```python
      p.ParamCmdLineArg("gray-scott", "settings", 1, ["settings.json"])
      ```
      means that the first parameter of "gray-scott" program, that in the campaign given a name "settings", has a value
      "settings.json". Notice that the value is given as a list suggesting that you can list here all possible values
      of the first positional parameter with which you want to launch the corresponding executable.
    + <b>ParamConfig</b> allows to deal with `*json` or `*ini` parameter files.
      For example
      ```python
      p.ParamConfig("gray-scott", "L", "settings.json", "L", [32, 64])
      ```
      means that parameter "L" from "settings.json" (that "gray-scott" reads) can take values 32 and 64 and is also called "L" inside the campaign.
    + <b>ParamRunner</b> allows to specify resources for each program.
      For example
      ```python
      p.ParamRunner('gray-scott', 'nprocs', [4] )
      ```
      means that "gray-scott" would use 4 MPI ranks. As with any other cheetah parameters, one can specify several
      values for such parameters as well which is needed for codesign studies.
    + When one creates a campaign with the above specification file, the campaign will have 4 experiments (2x2 parameter combinations).
    + Notice that parameters are given internal campaign name because one can use lambda functions to generate dependencies
      between different parameters and define <b>derived parameters</b> by using expressions with names of other parameters.
    + <b>SweepGroup</b> has <b>run_repetitions=2</b> parameter that says that each experiment should be repeated twice.
    + In each experiment specified above there are two MPI jobs running:
      - ["gray-scott"](https://github.com/pnorbert/adiosvm/tree/master/Tutorial/gray-scott) simulation  generates values on 3D grid at each time step, it uses 4 MPI ranks,
      - "compression" program at each time step
      	reads this 3D volume, compresses it with one of the compressors, such as [SZ](https://www.mcs.anl.gov/~shdi/download/sz-download.html),
	[ZFP](https://github.com/LLNL/zfp), [MGARD](https://github.com/CODARcode/MGARD.git), decompresses it back,
	runs [Z-Checker](https://github.com/CODARcode/Z-checker) and [FTK](https://github.com/CODARcode/ftk) on the original and lossy data to decide on the quality of
      	the compression; this job has 1 MPI rank.
  - Although ADIOS2 is not part of Cheetah, to understand how the programs, launched in parallel by Cheetah, communicate with each other,
    let us look into <b>adios2.xml</b> configuration file:
    ```xml
    <?xml version="1.0"?>
    <adios-config>
      <io name="SimulationOutput">
        <engine type="SST">
          <parameter key="RendezvousReaderCount" value="1"/>
          <parameter key="QueueLimit" value="15"/>
          <parameter key="QueueFullPolicy" value="Block"/>
        </engine>
      </io>
      <io name="CompressionOutput">
        <engine type="BPFile">
          <parameter key="RendezvousReaderCount" value="1"/>
          <parameter key="QueueLimit" value="15"/>
          <parameter key="QueueFullPolicy" value="Discard"/>
        </engine>
      </io>
    </adios-config>
    ```
    + Inside Gray-Scott program, using ADIOS2 API, a user opens <b>SimulationOutput</b> stream and writes to it at each time step
      without knowing what I/O backend is used: BP file, HDF5 file,
      network socket (SST, SSC), etc.
    + Inside compression program, using ADIOS2 API, a user  opens <b>SimulationOutput</b> stream and reads from it at each time step
    + Of course, both the reader and the writer should be told to use the same <b>adios2.xml</b> file.
    + The above XML file specifies that  <b>SimulationOutput</b> uses <b>SST</b> engine (network socket) and that a producer should block until somebody reads its output.
    + <b>CompressionOutput</b> stream is used by compression program to write its output into BP file (ADIOS2's native output format).
    + Engines can be changed in XML file without rebuilding the programs.

## Directory structure of the campaign
* Once one creates a campaign with the above specification file, the following directory structure is created:
  ```bash
  <campaign dir>/<username>/<campaign name>/run-<X>.iteration-<Y>
  ```
  - Here <b>campaign dir</b> is what is specified with `-o` option when running `cheetah.py create-campaign -o <campaign dir> ...`.
  - <b>campaign name</b> is what is set as <b>name</b> field in the specification file
  - <b>X</b> enumerates all possible combinations of parameters
  - <b>Y</b> goes over <b>run_repetitions</b> from <b>SweepGroup</b>
* Inside each <b>run-\<X\>.iteration-\<Y\></b> there are subdirectories corresponding to the programs in the experiment. For example, for the above speficiation file, there
  are <b>gray-scott</b> and <b>compression</b> directories. There are also corresponding subdirectories with <b>codar.cheetah.tau-</b> prefix, which corresponds to the runs
  of the programs in which tau was used for profiling. Each subdirectory might contain configuration, launch, monitor, log files appropriate for the corresponding level.
  - `<campaign dir>/<username>` has <b>run-all.sh</b> that can be used to start the whole campaign.
  - `<campaign dir>/<username>/<campaign name>` has <b>cancel.sh</b> and <b>status.sh</b> that can be used to stop or monitor the campaign.
  - `<campaign dir>/<username>/<campaign name>/run-<X>.iteration-<Y>` has the parameter files for this particular run.
  - The corresponding subdirectories for the particular programs in the experiment would also contain their parameter files and the logs would be created there for stdout, stderr,
    return status, walltime, etc.
    
## Examples
* For more examples of using Cheetah, see
  - [Calculating Euler's number](https://github.com/CODARcode/cheetah/tree/dev/examples/eulers_number)
  - [The Gray-Scott Reaction-Diffusion Benchmark](https://github.com/CODARcode/cheetah/tree/dev/examples/gray-scott)
  - [Brusselator](https://github.com/CODARcode/cheetah/tree/dev/examples/brusselator)
  - [Gray-Scott with compression, Z-Checker and FTK](https://github.com/CODARcode/cheetah/tree/dev/examples/gray-scott-compression)

   