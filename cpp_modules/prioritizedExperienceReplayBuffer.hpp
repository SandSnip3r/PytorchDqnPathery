#ifndef PRIORITIZED_EXPERIENCE_REPLAY_BUFFER_HPP_
#define PRIORITIZED_EXPERIENCE_REPLAY_BUFFER_HPP_

#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

template<typename T>
class PrioritizedExperienceReplayBuffer {
private:
  struct CircularBufferItem {
    CircularBufferItem() = default;
    CircularBufferItem(const T &item) : item(item) {}
    T item;
    std::size_t heapIndex;
  };
  struct HeapItem {
    HeapItem(double priority, std::size_t circularBufferIndex) : priority(priority), circularBufferIndex(circularBufferIndex) {}
    double priority;
    std::size_t circularBufferIndex;
  };
public:
  struct SampledItem {
    SampledItem(const T &item, std::size_t itemId, double weight) : item(item), itemId(itemId), weight(weight) {}
    T item;
    std::size_t itemId;
    double weight;
  };

  PrioritizedExperienceReplayBuffer(int capacity, int sampleSize, double alpha) : capacity_(capacity), sampleSize_(sampleSize), alpha_(alpha) {
    bucketBounds_.resize(sampleSize_);
    maxHeap_.reserve(capacity_);
  }

  void push(const T &item, double priority) {
    const std::size_t beforeSize = size();
    if (atCapacity_) {
      const std::size_t heapIndexToOverwrite = circularBuffer_[circularBufferNextIndex_].heapIndex;
      circularBuffer_[circularBufferNextIndex_].item = item;
      maxHeap_[heapIndexToOverwrite] = HeapItem(priority, circularBufferNextIndex_);
      moveNewHeapDataIntoPlace(heapIndexToOverwrite);
    } else {
      circularBuffer_[circularBufferNextIndex_] = CircularBufferItem(item);
      // Not at capacity
      maxHeap_.emplace_back(priority, circularBufferNextIndex_);
      circularBuffer_[circularBufferNextIndex_].heapIndex = maxHeap_.size()-1;
      moveNewHeapDataIntoPlace(maxHeap_.size()-1);
      if (circularBufferNextIndex_ == capacity_-1) {
        atCapacity_ = true;
      }
    }
    circularBufferNextIndex_ = (circularBufferNextIndex_+1) % capacity_;

    // If the size of the buffer changed, recompute bucket bounds.
    const std::size_t currentSize = size();
    if (currentSize > beforeSize) {
      prioritySum_ += std::pow(1.0/currentSize, alpha_);
      // Don't bother computing buckets until we have at least enough items for a sample.
      if (currentSize >= sampleSize_) {
        double cumulativeSum = 0.0;
        std::size_t currentBucketIndex = 0;
        for (std::size_t i=0; i<currentSize; ++i) {
          const double priority = std::pow(1.0/(i+1), alpha_);
          const double probability = priority/prioritySum_;
          cumulativeSum += probability;
          if (cumulativeSum >= static_cast<double>(currentBucketIndex+1) / sampleSize_) {
            // New bucket boundary
            bucketBounds_.at(currentBucketIndex) = i+1;
            ++currentBucketIndex;
            if (currentBucketIndex == sampleSize_-1) {
              // Add the last bucket-end
              bucketBounds_.back() = currentSize;
              break;
            }
          }
        }
      }
    }
  }

  // Returns a list of pairs of objects & indices, for updating priorities.
  std::vector<SampledItem> sample(double beta) {
    if (size() < sampleSize_) {
      throw std::runtime_error("Not enough items to sample");
    }
    std::vector<SampledItem> result;
    result.reserve(sampleSize_);
    std::size_t start=0;
    for (std::size_t end : bucketBounds_) {
      std::uniform_int_distribution<std::size_t> dist(start, end-1);
      const std::size_t sampledIndex = dist(eng_);
      const std::size_t circularBufferIndex = maxHeap_.at(sampledIndex).circularBufferIndex;
      // Calculate the weight associated with this item.
      const double importanceSamplingWeight = std::pow((1.0 / size() * (end-start)), beta);
      // The item ID in the `SampledItem` is an index into the `maxHeap_`.
      result.emplace_back(circularBuffer_[circularBufferIndex].item, sampledIndex, importanceSamplingWeight);
      start = end;
    }
    // Normalize the weights
    double maxWeight = 0.0;
    for (const SampledItem &item : result) {
      maxWeight = std::max(maxWeight, item.weight);
    }
    for (SampledItem &item : result) {
      item.weight /= maxWeight;
    }
    return result;
  }

  void updatePriority(std::size_t itemId, double newPriority) {
    maxHeap_.at(itemId).priority = newPriority;
    moveNewHeapDataIntoPlace(itemId);
  }

  std::size_t size() const {
    if (atCapacity_) {
      return circularBuffer_.size();
    } else {
      return circularBufferNextIndex_;
    }
  }

private:
  // We have two data structures:
  //  1. A circular buffer for FIFO; std::vector (std::deque is not good, because poping shifts over all indices in the other data structure)
  //  2. An array-based binary-max-heap: std::vector
  //
  // When at capcity, we remove the oldest thing from the FIFO then need an index into the heap to go remove it from there also. When removing it from the heap, we cannot have a hole, so we insert the new item there and then bubble it up or down. Meanwhile we need to update all index tracking in the circular buffer.
  const std::size_t capacity_;
  const int sampleSize_;
  const double alpha_;
  std::vector<CircularBufferItem> circularBuffer_{std::vector<CircularBufferItem>(capacity_)};
  std::vector<HeapItem> maxHeap_;
  std::size_t circularBufferNextIndex_{0};
  bool atCapacity_{false};
  std::mt19937 eng_{0/* createRandomEngine() */}; // TODO: Take seed for random engine in constructor.

  std::vector<std::size_t> bucketBounds_;
  double prioritySum_{0.0};

  void moveNewHeapDataIntoPlace(std::size_t currentIndex) {
    const auto beforeIndex = currentIndex;
    // Maybe bubble up
    while (currentIndex > 0) {
      // Is our parent lower priority?
      const std::size_t parentIndex = (currentIndex-1) / 2;
      if (maxHeap_[currentIndex].priority > maxHeap_[parentIndex].priority) {
        // Need to to bubble up
        std::swap(maxHeap_[parentIndex], maxHeap_[currentIndex]);
        // Also, swap the indices in the circular buffer.
        std::swap(circularBuffer_[maxHeap_[parentIndex].circularBufferIndex].heapIndex,
                  circularBuffer_[maxHeap_[currentIndex].circularBufferIndex].heapIndex);
        currentIndex = parentIndex;
      } else {
        // Not greater, done
        break;
      }
    }
    // Maybe bubble down
    while (1) {
      const std::size_t child1Index = currentIndex*2 + 1;
      const std::size_t child2Index = currentIndex*2 + 2;
      if (child1Index >= circularBuffer_.size()) {
        // No children
        break;
      }
      if (child2Index >= circularBuffer_.size()) {
        // Only have one child
        if (maxHeap_[currentIndex].priority < maxHeap_[child1Index].priority) {
          // Need to bubble down
          std::swap(maxHeap_[currentIndex], maxHeap_[child1Index]);
          // Also, swap the indices in the circular buffer.
          std::swap(circularBuffer_[maxHeap_[currentIndex].circularBufferIndex].heapIndex,
                    circularBuffer_[maxHeap_[child1Index].circularBufferIndex].heapIndex);
          currentIndex = child1Index;
        } else {
          // Not less, done
          break;
        }
      } else {
        // Have two children; must switch with the larger one, if any.
        const std::size_t largerChildIndex = (maxHeap_[child1Index].priority > maxHeap_[child2Index].priority) ? child1Index : child2Index;
        if (maxHeap_[currentIndex].priority < maxHeap_[largerChildIndex].priority) {
          // Need to bubble down
          std::swap(maxHeap_[currentIndex], maxHeap_[largerChildIndex]);
          // Also, swap the indices in the circular buffer.
          std::swap(circularBuffer_[maxHeap_[currentIndex].circularBufferIndex].heapIndex,
                    circularBuffer_[maxHeap_[largerChildIndex].circularBufferIndex].heapIndex);
          currentIndex = largerChildIndex;
        } else {
          // Not less, done
          break;
        }
      }
    }
  }
};

#endif // PRIORITIZED_EXPERIENCE_REPLAY_BUFFER_HPP_