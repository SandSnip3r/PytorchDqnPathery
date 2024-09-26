#include "prioritizedExperienceReplayBuffer.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // For binding std::vector

namespace py = pybind11;

template<typename T>
void declare_prioritized_experience_replay_buffer(py::module &m, const std::string &type) {
  using Buffer = PrioritizedExperienceReplayBuffer<T>;

  // Bind the ItemAndDataIndex struct
  py::class_<typename Buffer::ItemAndDataIndex>(m, ("ItemAndDataIndex" + type).c_str())
    .def_readonly("item", &Buffer::ItemAndDataIndex::item)
    .def_readonly("dataIndex", &Buffer::ItemAndDataIndex::dataIndex);
  
  py::class_<Buffer>(m, ("PrioritizedExperienceReplayBuffer" + type).c_str())
    .def(py::init<int, int, double>(), py::arg("capacity"), py::arg("sampleSize"), py::arg("alpha"))
    .def("push", &Buffer::push, py::arg("item"), py::arg("priority"))
    .def("sample", &Buffer::sample)
    .def("updatePriority", &Buffer::updatePriority, py::arg("dataIndexOfPriorityToUpdate"), py::arg("newPriority"))
    .def("__len__", &Buffer::size);
}

PYBIND11_MODULE(prioritized_buffer, m) {
  // For primitive types
  // declare_prioritized_experience_replay_buffer<int>(m, "Int");
  // declare_prioritized_experience_replay_buffer<float>(m, "Float");
  // declare_prioritized_experience_replay_buffer<std::string>(m, "String");

  // For arbitrary Python objects
  declare_prioritized_experience_replay_buffer<py::object>(m, "Object");
}