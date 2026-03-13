# DataRequest API Tutorial

This document walks through the use of the `DataRequest` helper class located
in `src/libs/data_request.py`.  The class wraps several endpoints of the
public ANE RSM API and adds client-side caching to speed up repeated
analyses.

## Getting Started

First, ensure your workspace has a logger configured. The project provides a
convenience wrapper in `src/cfg.py`, but you can also pass any
`logging.Logger` instance.

```python
import cfg
from libs.data_request import DataRequest

log = cfg.set_logger()
dr = DataRequest(log=log, base_url=cfg.API_URL)
```

The `base_url` parameter defaults to the production API.  During development
you may override it (for example, pointing at a mock server) without
changing any other code.

## Inspect Campaign Parameters

Campaign metadata includes schedule and SDR configuration.  Use
`get_campaign_params` to retrieve a `CampaignParams` dataclass with
handy attributes and a computed `pxx_len` value that is used elsewhere in
the analysis routines.

```python
info = dr.get_campaign_params(176)
print(info.name)              # human-readable campaign name
print(info.schedule.start_date)
print(info.config.sample_rate_hz)
print(info.pxx_len)           # computed FFT length
```

The method is safe against malformed responses; if the API returns missing
or non-numeric fields the code falls back to defaults and logs the event.

## Downloading Signals

### Single Node (Realtime)

For quick lookups you can fetch the realtime snapshot for a node by using
`get_realtime_signal`.

```python
realtime = dr.get_realtime_signal(1)
print(realtime.pxx.shape)
``` 

### Bulk Historical Data with Caching

Often you will want to pull entire campaigns of data for multiple nodes.
`load_campaigns_and_nodes` will iterate through the provided campaigns and
node IDs, download the paginated results using :meth:`get_api_signals`, and
store the results in a local ``.cache`` directory.  Returned values are
:class:`pandas.DataFrame` objects keyed by campaign label and node name.

```python
campaigns = {"FM original": 176}
nodes = list(range(1, 11))

all_data = dr.load_campaigns_and_nodes(campaigns, nodes)

# access network for campaign "FM original" and node 5
df_node5 = all_data["FM original"]["Node5"]
```

Subsequent calls with the same campaign/node combination will load the
DataFrame from disk immediately, bypassing the network entirely.

## Advanced Notes

* The `pxx_len` computation mirrors the projects signal‑processing logic
  and chooses the next power-of-two FFT length based on the RBW and sample
  rate.  You generally only need it when reconstructing frequency axes.
* The `get_api_signals` method includes a progress bar thanks to ``tqdm`` and
  will stop cleanly if the API returns an error or times out.  Logged errors
  appear on the console but don't raise exceptions; partial data is returned
  instead.
* You can access the raw request cache via ``dr.cache`` if you need to purge
  or inspect entries.

## Example in Notebook

See `src/example-params-campaign.ipynb` for an interactive demonstration of
the basic calls described above.
