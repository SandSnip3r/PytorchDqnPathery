#ifndef PRIORITIZED_EXPERIENCE_REPLAY_BUFFER_HPP_
#define PRIORITIZED_EXPERIENCE_REPLAY_BUFFER_HPP_

#include <cmath>
#include <deque>
#include <iostream>
#include <vector>

#include <algorithm>
#include <array>
#include <functional>
#include <random>

template<typename T>
class PrioritizedExperienceReplayBuffer {
public:
  struct SampledItem {
    SampledItem(const T &item, std::size_t dataIndex, double weight) : item(item), dataIndex(dataIndex), weight(weight) {}
    T item;
    std::size_t dataIndex;
    double weight;
  };
private:
  struct ItemAndSortedIndex {
    ItemAndSortedIndex(const T &item, std::size_t sortedIndex) : item(item), sortedIndex(sortedIndex) {}
    T item;
    std::size_t sortedIndex;
  };
  struct PriorityAndDataIndex {
    PriorityAndDataIndex(double priority, std::size_t dataIndex) : priority(priority), dataIndex(dataIndex) {}
    double priority;
    std::size_t dataIndex;
  };
public:
  PrioritizedExperienceReplayBuffer(int capacity, int sampleSize, double alpha) : capacity_(capacity), sampleSize_(sampleSize), alpha_(alpha) {
    bucketBounds_.resize(sampleSize_);
  }

  void push(const T &item, double priority) {
    // std::cout << std::endl;
    // printLists();
    const std::size_t beforeSize = dataBuffer_.size();
    // If the buffer is at capacity, remove the oldest item
    if (beforeSize == capacity_) {
      // std::cout << "At capacity (" << capacity_ << "), popping item \"" << dataBuffer_.front().item << '"' << std::endl;
      const std::size_t indexOfItemInSortedList = dataBuffer_.front().sortedIndex;
      dataBuffer_.pop_front();
      // Remove this item from the sorted list
      // std::cout << "  Removing item " << indexOfItemInSortedList << " from sorted list" << std::endl;
      sortedIndices_.erase(sortedIndices_.begin()+indexOfItemInSortedList);
      // For everything in the dataBuffer_, if the index is >= indexOfItemInSortedList, then decrement
      int i=0;
      for (ItemAndSortedIndex &itemAndIndex : dataBuffer_) {
        if (itemAndIndex.sortedIndex > indexOfItemInSortedList) {
          --itemAndIndex.sortedIndex;
          // std::cout << "    Shifting index of item in data buffer at index " << i << " to " << itemAndIndex.sortedIndex << std::endl;
        }
        ++i;
      }
      // For everything in the sortedIndices_, decrement the index
      for (PriorityAndDataIndex &priorityAndIndex : sortedIndices_) {
        --priorityAndIndex.dataIndex;
      }
    }

    // Push the item to the back of the data buffer
    // std::cout << "Pushing item \"" << item << "\" with priority " << priority << std::endl;
    dataBuffer_.emplace_back(item, 0);

    // Insert in-order into the sorted list
    const auto insertBeforeIterator = std::upper_bound(sortedIndices_.begin(), sortedIndices_.end(), priority, [](const double priorityOfItemToInsert, const auto &itemInSortedList) {
      return priorityOfItemToInsert > itemInSortedList.priority;
    });
    const auto iteratorOfEmplacedItem = sortedIndices_.emplace(insertBeforeIterator, priority, dataBuffer_.size()-1);
    const std::size_t sortedInsertedIndex = std::distance(sortedIndices_.begin(), iteratorOfEmplacedItem);

    // Update all indices in the data buffer
    for (ItemAndSortedIndex &itemAndIndex : dataBuffer_) {
      if (itemAndIndex.sortedIndex >= sortedInsertedIndex) {
        // Shifted right by 1
        ++itemAndIndex.sortedIndex;
      }
    }

    // Set the index of the item in the data buffer to point to the inserted item in the sorted list.
    dataBuffer_.back().sortedIndex = sortedInsertedIndex;
    // std::cout << "  Inserted into index " << sortedInsertedIndex << " in sorted list" << std::endl;

    // If the size of the buffer changed, recompute bucket bounds.
    if (dataBuffer_.size() > beforeSize) {
      // std::cout << "  Buffer size changed, recomputing bucket bounds" << std::endl;
      // std::cout << "  Updating priority sum from " << prioritySum_;
      prioritySum_ += std::pow(1.0/dataBuffer_.size(), alpha_);
      // std::cout << " to " << prioritySum_ << std::endl;

      if (dataBuffer_.size() >= sampleSize_) {
        // std::cout << "  Recalculating buckets" << std::endl;
        double cumulativeSum = 0.0;
        std::size_t currentBucketIndex = 0;
        for (std::size_t i=0; i<dataBuffer_.size(); ++i) {
          const double priority = std::pow(1.0/(i+1), alpha_);
          const double probability = priority/prioritySum_;
          cumulativeSum += probability;
          // std::cout << "    Cumulative sum: " << cumulativeSum << " (vs " << static_cast<double>(currentBucketIndex+1) / sampleSize_ << ")" << std::endl;
          if (cumulativeSum >= static_cast<double>(currentBucketIndex+1) / sampleSize_) {
            bucketBounds_.at(currentBucketIndex) = i+1;
            ++currentBucketIndex;
            if (currentBucketIndex == sampleSize_-1) {
              // Add one last bucket-end
              bucketBounds_.back() = dataBuffer_.size();
              break;
            }
          }
        }
        // std::cout << "    Computed bucket bounds as [";
        // for (const auto i : bucketBounds_) {
        //   std::cout << i << ',';
        // }
        // std::cout << ']' << std::endl;
      } else {
        // std::cout << "  Do not have enough items for sampling, skipping computation of bucket bounds" << std::endl;
      }
    }
  }

  // Returns a list of pairs of objects & indices, for updating priorities.
  std::vector<SampledItem> sample(double beta) {
    if (dataBuffer_.size() < sampleSize_) {
      throw std::runtime_error("Not enough items to sample");
    }
    // std::cout << "Sampling " << sampleSize_ << " items" << std::endl;
    // printLists(1);
    std::vector<SampledItem> result;
    result.reserve(sampleSize_);
    std::size_t start=0;
    // std::cout << "  Bucket bounds: [";
    // for (const auto b : bucketBounds_) {
    //   std::cout << b << ',';
    // }
    // std::cout << ']' << std::endl;
    for (std::size_t end : bucketBounds_) {
      // std::cout << "    Sampling from [" << start << ',' << end << ")" << std::endl;
      std::uniform_int_distribution<std::size_t> dist(start, end-1);
      const std::size_t sampledSortedIndex = dist(eng_);
      // std::cout << "    sampledSortedIndex: " << sampledSortedIndex << std::endl;
      const std::size_t sampledDataIndex = sortedIndices_.at(sampledSortedIndex).dataIndex;
      // std::cout << "    sampledDataIndex: " << sampledDataIndex << std::endl;
      const auto &dataBufferItem = dataBuffer_[sampledDataIndex];
      // std::cout << "    Sampled item \"" << dataBufferItem.item << "\" from data index " << sampledDataIndex << std::endl;
      // Calculate the weight associated with this item.
      const double importanceSamplingWeight = std::pow((1.0 / dataBuffer_.size() * (end-start)), beta);
      result.emplace_back(dataBufferItem.item, sampledDataIndex, importanceSamplingWeight);
      start = end;
    }
    return result;
  }

  void updatePriority(std::size_t dataIndexOfPriorityToUpdate, double newPriority) {
    const std::size_t sortedIndex = dataBuffer_.at(dataIndexOfPriorityToUpdate).sortedIndex;
    sortedIndices_.at(sortedIndex).priority = newPriority;
    // Find new position of item
    std::size_t finalIndex = sortedIndex;
    // Only one of these while loops will execute.
    while (finalIndex < sortedIndices_.size()-1 && sortedIndices_.at(sortedIndex).priority < sortedIndices_.at(finalIndex+1).priority) {
      ++finalIndex;
    }
    while (finalIndex > 0 && sortedIndices_.at(sortedIndex).priority > sortedIndices_.at(finalIndex-1).priority) {
      --finalIndex;
    }
    if (finalIndex == sortedIndex) {
      // Already where it should be; nothing to do.
      return;
    }
    // Remove our item from the list. This will shift everything after it to the left by 1.
    const PriorityAndDataIndex sortedItem = sortedIndices_.at(sortedIndex);
    sortedIndices_.erase(sortedIndices_.begin()+sortedIndex);
    if (finalIndex > sortedIndex) {
      // Update the indices of all permanently moved data buffers.
      for (std::size_t i=sortedIndex; i<finalIndex; ++i) {
        dataBuffer_.at(sortedIndices_.at(i).dataIndex).sortedIndex = i;
      }
      // Insert ourt item into the new position.
      sortedIndices_.insert(sortedIndices_.begin()+finalIndex, sortedItem);
    } else if (finalIndex < sortedIndex) {
      // Insert ourt item into the new position.
      sortedIndices_.insert(sortedIndices_.begin()+finalIndex, sortedItem);
      // Update the indices of all permanently moved data buffers.
      for (std::size_t i=finalIndex+1; i<=sortedIndex; ++i) {
        dataBuffer_.at(sortedIndices_.at(i).dataIndex).sortedIndex = i;
      }
    }
    // Also update the index of our item
    dataBuffer_.at(sortedItem.dataIndex).sortedIndex = finalIndex;
  }

  std::size_t size() const {
    return dataBuffer_.size();
  }
private:
  const int capacity_;
  const int sampleSize_;
  const double alpha_;
  std::mt19937 eng_{0/* createRandomEngine() */}; // TODO: Take seed for random engine in constructor.

  std::deque<ItemAndSortedIndex> dataBuffer_;
  // Keep a descending sorted list based on priorities.
  std::vector<PriorityAndDataIndex> sortedIndices_; // TODO: Maybe a deque would be faster here.
  // NOTE: Data should live in the deque, rather than the sorted array, because there will be a lot of shifting inserts in the sorted array.

  std::vector<std::size_t> bucketBounds_;
  double prioritySum_{0.0};

  // static std::mt19937 createRandomEngine() {
  //   std::random_device rd{seed};
  //   std::array<int, std::mt19937::state_size> seed_data;
  //   std::generate_n(seed_data.data(), seed_data.size(), std::ref(rd));
  //   std::seed_seq seq(std::begin(seed_data), std::end(seed_data));
  //   return std::mt19937(seq);
  // }

// public:
//   void printLists(int indentLevel=0) const {
//     std::cout << std::string(indentLevel*2, ' ') << "dataBuffer_: [\n";
//     for (const ItemAndSortedIndex &item : dataBuffer_) {
//       std::cout << std::string((indentLevel+1)*2, ' ') << '{' << '"' << item.item << "\"," << item.sortedIndex << "},\n";
//     }
//     std::cout << std::string(indentLevel*2, ' ') << "]\n";
//     std::cout << std::string(indentLevel*2, ' ') << "sortedIndices_: [\n";
//     for (const PriorityAndDataIndex &item : sortedIndices_) {
//       std::cout << std::string((indentLevel+1)*2, ' ') << '{' << item.priority << ',' << item.dataIndex << "},\n";
//     }
//     std::cout << std::string(indentLevel*2, ' ') << "]\n";
//   }
  // void printSortedItemsWithPriorities(int indentLevel=0) const {
  //   for (std::size_t i=0; i<sortedIndices_.size(); ++i) {
  //     std::cout << std::string(indentLevel*2, ' ') << '"' << dataBuffer_.at(sortedIndices_.at(i).dataIndex).item << "\" " << sortedIndices_.at(i).priority << std::endl;
  //   }
  // }
};

#endif // PRIORITIZED_EXPERIENCE_REPLAY_BUFFER_HPP_