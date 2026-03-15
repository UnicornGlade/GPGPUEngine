#include <gtest/gtest.h>

#include <libbase/lru_cache.h>

TEST(lru_cache, storesAndReadsValues)
{
	libbase::LruCache<int, int> cache(2);

	EXPECT_EQ(cache.capacity(), 2);
	EXPECT_TRUE(cache.empty());
	EXPECT_FALSE(cache.contains(1));

	cache.put(1, 10);

	ASSERT_EQ(cache.size(), 1);
	ASSERT_TRUE(cache.contains(1));
	ASSERT_NE(cache.get(1), nullptr);
	EXPECT_EQ(*cache.get(1), 10);
}

TEST(lru_cache, updatesRecencyOnHit)
{
	libbase::LruCache<int, int> cache(2);
	cache.put(1, 10);
	cache.put(2, 20);

	ASSERT_NE(cache.get(1), nullptr);
	cache.put(3, 30);

	EXPECT_TRUE(cache.contains(1));
	EXPECT_FALSE(cache.contains(2));
	EXPECT_TRUE(cache.contains(3));
	EXPECT_EQ(cache.keysMostRecentFirst(), (std::vector<int>{3, 1}));
}

TEST(lru_cache, overwritesExistingValueWithoutGrowing)
{
	libbase::LruCache<int, int> cache(2);
	cache.put(1, 10);
	cache.put(2, 20);

	cache.put(1, 15);

	ASSERT_EQ(cache.size(), 2);
	ASSERT_NE(cache.get(1), nullptr);
	EXPECT_EQ(*cache.get(1), 15);
	EXPECT_EQ(cache.keysMostRecentFirst(), (std::vector<int>{1, 2}));
}

TEST(lru_cache, evictsLeastRecentlyUsed)
{
	libbase::LruCache<int, int> cache(2);
	cache.put(1, 10);
	cache.put(2, 20);
	cache.put(3, 30);

	EXPECT_FALSE(cache.contains(1));
	EXPECT_TRUE(cache.contains(2));
	EXPECT_TRUE(cache.contains(3));
	EXPECT_EQ(cache.keysMostRecentFirst(), (std::vector<int>{3, 2}));
}

TEST(lru_cache, supportsCapacityOne)
{
	libbase::LruCache<int, int> cache(1);
	cache.put(1, 10);
	cache.put(2, 20);

	EXPECT_FALSE(cache.contains(1));
	EXPECT_TRUE(cache.contains(2));
	EXPECT_EQ(cache.keysMostRecentFirst(), (std::vector<int>{2}));
}

TEST(lru_cache, capacityZeroBehavesAsDisabledCache)
{
	libbase::LruCache<int, int> cache(0);
	cache.put(1, 10);

	EXPECT_EQ(cache.size(), 0);
	EXPECT_FALSE(cache.contains(1));
	EXPECT_EQ(cache.get(1), nullptr);
	EXPECT_TRUE(cache.keysMostRecentFirst().empty());
}
