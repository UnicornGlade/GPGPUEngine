#pragma once

#include <cstddef>
#include <list>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include <libbase/runtime_assert.h>

namespace libbase {

template <typename Key, typename Value, typename Hash = std::hash<Key>, typename KeyEqual = std::equal_to<Key>>
class LruCache {
public:
	explicit LruCache(size_t capacity)
		: capacity_(capacity)
	{}

	size_t capacity() const
	{
		return capacity_;
	}

	size_t size() const
	{
		return entries_by_key_.size();
	}

	bool empty() const
	{
		return entries_by_key_.empty();
	}

	bool contains(const Key &key) const
	{
		return entries_by_key_.count(key) > 0;
	}

	void clear()
	{
		recency_.clear();
		entries_by_key_.clear();
	}

	void erase(const Key &key)
	{
		auto it = entries_by_key_.find(key);
		if (it == entries_by_key_.end()) {
			return;
		}

		recency_.erase(it->second.recency_it);
		entries_by_key_.erase(it);
	}

	Value *get(const Key &key)
	{
		auto it = entries_by_key_.find(key);
		if (it == entries_by_key_.end()) {
			return nullptr;
		}

		touch(it);
		return &it->second.value;
	}

	const Value *get(const Key &key) const
	{
		auto it = entries_by_key_.find(key);
		if (it == entries_by_key_.end()) {
			return nullptr;
		}

		return &it->second.value;
	}

	template <typename ValueLike>
	Value &put(const Key &key, ValueLike &&value)
	{
		if (capacity_ == 0) {
			discarded_value_ = Value(std::forward<ValueLike>(value));
			return *discarded_value_;
		}

		auto it = entries_by_key_.find(key);
		if (it != entries_by_key_.end()) {
			it->second.value = std::forward<ValueLike>(value);
			touch(it);
			return it->second.value;
		}

		recency_.push_front(key);
		Entry entry = {std::forward<ValueLike>(value), recency_.begin()};
		auto [inserted_it, inserted] = entries_by_key_.emplace(key, std::move(entry));
		rassert(inserted, 2026031511472700001);

		evictIfNeeded();
		return inserted_it->second.value;
	}

	std::vector<Key> keysMostRecentFirst() const
	{
		std::vector<Key> keys;
		keys.reserve(recency_.size());
		for (const Key &key: recency_) {
			keys.push_back(key);
		}
		return keys;
	}

private:
	struct Entry {
		Value value;
		typename std::list<Key>::iterator recency_it;
	};

	using Map = std::unordered_map<Key, Entry, Hash, KeyEqual>;
	using MapIterator = typename Map::iterator;

	void touch(const MapIterator &it)
	{
		recency_.splice(recency_.begin(), recency_, it->second.recency_it);
		it->second.recency_it = recency_.begin();
	}

	void evictIfNeeded()
	{
		rassert(capacity_ > 0, 2026031511472700002);
		while (entries_by_key_.size() > capacity_) {
			rassert(!recency_.empty(), 2026031511472700003);
			auto last_it = std::prev(recency_.end());
			const Key &evicted_key = *last_it;
			size_t erased = entries_by_key_.erase(evicted_key);
			rassert(erased == 1, 2026031511472700004);
			recency_.erase(last_it);
		}
	}

	size_t capacity_;
	std::list<Key> recency_;
	Map entries_by_key_;
	std::optional<Value> discarded_value_;
};

}
