from run_def import reconstruction_job

from validation.implementations.attacks.attack_def import (
    attack_preset_def,
)

from validation.implementations.attacks.dlg import dlg


def idlg() -> reconstruction_job:

    # build the original DLG configuration
    attack = dlg()

    # iDLG analytically infers the label
    # rather than optimizing unknown labels
    attack.label_policy = "idlg"

    return attack


IDLG_PRESET = attack_preset_def(
    name="idlg",
    builder=idlg,
    max_iterations=300,
    faithful=True,
)