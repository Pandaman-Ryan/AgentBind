import subprocess

for idx in range(40,50):
    workdir = "/storage/pandaman/project/AgentBind-IMPACT/tmp-c%d/" %idx
    resultdir = "/storage/pandaman/project/AgentBind-IMPACT/results-c%d/" %idx
    cmd = "python AgentBind.py --workdir %s --resultdir %s" %(workdir, resultdir)
    subprocess.check_call(cmd, shell=True)

