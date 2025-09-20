from nipype.interfaces import spm

MATLAB_RUNTIME = '/home/rongfei/MATLAB/runtime/R2024b'
SPM_START_SCRIPT = '/home/rongfei/Builds/spm12/spm_standalone/run_spm25.sh'

MCR_COMMAND = f'{SPM_START_SCRIPT} {MATLAB_RUNTIME} script'
print("MCR_COMMAND: ", MCR_COMMAND)

spm.SPMCommand.set_mlab_paths(MCR_COMMAND, use_mcr=True)
print(spm.SPMCommand().version)