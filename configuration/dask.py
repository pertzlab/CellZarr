from dask.distributed import Client, LocalCluster

## Local Analysis
LOCAL_DASK_CLUSTER = True


def start_dask_cluster():
    """
    Start a Dask cluster based on the LOCAL_DASK_CLUSTER setting.
    If LOCAL_DASK_CLUSTER is True, a LocalCluster is started.
    Otherwise, a SLURMCluster is started.
    """
    if LOCAL_DASK_CLUSTER:
        cluster = LocalCluster()
    else:
        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(
            # n_workers=1,  # Number of workers to start
            cores=8,
            processes=4,
            memory="8GB",  # Memory per worker
            walltime="2:00:00",  # Walltime
            scheduler_options={"dashboard_address": f":{8787}"},
            log_directory="/tmp",
            nanny=True,
            worker_extra_args=["--nthreads", "1"],  # Only 1 thread used by Dask itself
        )
        cluster.scale(jobs=3)
    client = Client(cluster)
    client.amm.start()
    return cluster, client
