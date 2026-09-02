# Running on Nebius

[SkyPilot](https://docs.skypilot.ai) provisions the GPU, syncs the repo,
and runs the job. One YAML holds the whole run, so a rerun reproduces it.

## Credentials

```
uv tool install --with pip "skypilot[nebius]"
mkdir -p ~/.nebius
nebius iam get-access-token > ~/.nebius/NEBIUS_IAM_TOKEN.txt
nebius --format json iam whoami | jq -r '.user_profile.tenants[0].tenant_id' > ~/.nebius/NEBIUS_TENANT_ID.txt
sky check nebius
```

SkyPilot reads the IAM token and tenant id from those two files. The token
expires, so refresh the first file before a launch that fails to
authenticate. `sky check nebius` prints `Nebius: enabled` when it works.

## Launch

```
just sky-train
```

That runs `sky launch -c time-wizard sky/train.yaml`, which provisions one
L40S, installs uv, syncs the working directory, rebuilds the photo crops
from their seed, fine-tunes, and scores the adapter on the dev split.
Training data is never uploaded. `timewizard.photos` downloads the images
on the box, so the transfer is the repo alone.

Pass different training flags without editing the file:

```
sky launch -c time-wizard sky/train.yaml --env TRAIN_ARGS="--out runs/rendered --photos false --rendered 100000"
```

## Afterwards

```
just sky-fetch    # copy runs/ back from the box
just sky-down     # release the GPU
```

The cluster keeps billing while it is up, so take the adapter and stop it.
