set -x

export CHECK_CONFIG_CONSISTANCY=1
export HELION_DEBUG_DISTRIBUTED=1  # this one deprecates CHECK_CONFIG_CONSISTANCY

export HELION_AUTOTUNE_FOR_DISTRIBUTED_KERNEL=1

function commited_tests() {
    echo "M============== UNIT TEST ============="
    python test/test_distributed.py

    echo "M============== all reduce bias rmsnorm one shot =================="
    KERNEL_FILTER=one_shot HELION_FORCE_AUTOTUNE=1 torchrun --nproc-per-node=8 examples/distributed/allreduce_bias_rmsnorm.py

    echo "M============== all reduce bias rmsnorm two shot =================="
    KERNEL_FILTER=two_shot HELION_FORCE_AUTOTUNE=1 torchrun --nproc-per-node=8 examples/distributed/allreduce_bias_rmsnorm.py

    echo "M============= all reduce ============="
    HELION_FORCE_AUTOTUNE=1 torchrun --nproc-per-node=8 examples/distributed/all_reduce.py

    # debug why so slow: https://gist.github.com/shunting314/4c2d846e197a866719ce6545ac550ad1
    echo "M========== matmul reduce scatter =========="
    HELION_FORCE_AUTOTUNE=1 torchrun --nproc-per-node=8 examples/distributed/matmul_reduce_scatter.py

    echo "========== allgather matmul ============="
    HELION_FORCE_AUTOTUNE=1 torchrun --nproc-per-node=8 examples/distributed/all_gather_matmul.py
}

function candidate_tests() {
    echo
}

time commited_tests 2>&1 | tee /tmp/helion-dist.log
# candidate_tests
exit

############# SPLIT ##############
