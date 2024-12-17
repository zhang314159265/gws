from prepare_softmax_gpt2 import get_args, call as two_pass_call
from online_softmax import call as online_softmax_call
from common import benchmark, eager
import functools

def main():
    args = get_args()
    ref_max, ref_sum = eager(args[0])
    benchmark("TwoPassTrivial", two_pass_call, args, ref_max, ref_sum)

    two_pass_call_alt_loop_order = functools.partial(two_pass_call, alt_loop_order=True)
    benchmark("TwoPassAltLoopOrder", two_pass_call_alt_loop_order, args, ref_max, ref_sum)
    benchmark("OnlineSoftmaxBase", online_softmax_call, args, ref_max, ref_sum)
    online_softmax_call_opt = functools.partial(online_softmax_call, opt=True)
    benchmark("OnlineSoftmaxOpt", online_softmax_call_opt, args, ref_max, ref_sum)

if __name__ == "__main__":
    main()
