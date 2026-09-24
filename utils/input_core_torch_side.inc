// A driver's output Buf is one of the tensors the propagator bound (the
// skeletons allocate nothing), so the tensor it stands for is the one with the
// same data pointer.  An undefined Buf maps to an undefined tensor (Python
// sees None).  A Buf that matches nothing is a contract break, reported.
inline torch::Tensor tensor_of(const Buf& b, std::initializer_list<const torch::Tensor*> singles,
                               std::initializer_list<const std::vector<torch::Tensor>*> lists,
                               const char* what)
{
    if (!b.defined()) return torch::Tensor();
    for (const torch::Tensor* t : singles)
        if (t->defined() && t->data_ptr() == b.data_ptr()) return *t;
    for (const std::vector<torch::Tensor>* l : lists)
        for (const auto& t : *l)
            if (t.defined() && t.data_ptr() == b.data_ptr()) return t;
    SWEEP_CHECK(false, what, ": the driver handed back a buffer the propagator did not bind");
    return torch::Tensor();
}

inline ForwardOutput to_torch(const ForwardOutputCore& c, const ForwardInput& in)
{
    ForwardOutput out;
    out.wavefield = tensor_of(c.wavefield, {&in.u_allt_out}, {}, "ForwardOutput.wavefield");
    out.last_two = tensor_of(c.last_two, {&in.last_two}, {}, "ForwardOutput.last_two");
    out.record = tensor_of(c.record, {&in.record_out}, {}, "ForwardOutput.record");
    return out;
}

inline BackwardOutput to_torch(const BackwardOutputCore& c, const BackwardInput& in)
{
    BackwardOutput out;
    out.grads.reserve(c.grads.size());
    for (const auto& g : c.grads)
        out.grads.push_back(tensor_of(g, {}, {&in.grads_out}, "BackwardOutput.grads"));
    out.source_illumination = tensor_of(c.source_illumination, {}, {&in.illum_out}, "BackwardOutput.source_illumination");
    out.receiver_illumination = tensor_of(c.receiver_illumination, {}, {&in.illum_out}, "BackwardOutput.receiver_illumination");
    out.adcig = tensor_of(c.adcig, {&in.adcig_out}, {}, "BackwardOutput.adcig");
    return out;
}

inline RTMOutput to_torch(const RTMOutputCore& c, const BackwardInput& in)
{
    RTMOutput out;
    out.source_illumination = tensor_of(c.source_illumination, {}, {&in.illum_out}, "RTMOutput.source_illumination");
    out.receiver_illumination = tensor_of(c.receiver_illumination, {}, {&in.illum_out}, "RTMOutput.receiver_illumination");
    out.adcig = tensor_of(c.adcig, {&in.adcig_out}, {}, "RTMOutput.adcig");
    return out;
}
