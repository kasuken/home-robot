import unittest

from laika.house import Cliff, House, Person, Pose, Room, Thing, default_house, wall_with_gaps


def two_rooms() -> House:
    """Room "a" (0..300) and room "b" (300..600), both 200 deep, joined by a doorway at y 60..140."""
    rooms = [Room("a", 0, 0, 300, 200), Room("b", 300, 0, 600, 200)]
    walls = [
        *wall_with_gaps(0, 0, 600, 0),
        *wall_with_gaps(0, 200, 600, 200),
        *wall_with_gaps(0, 0, 0, 200),
        *wall_with_gaps(600, 0, 600, 200),
        *wall_with_gaps(300, 0, 300, 200, (60, 140)),
    ]
    return House(rooms, walls, cliffs=[Cliff("stairs", 500, 0, 600, 200)], start=Pose(50, 100, 0))


class WallTests(unittest.TestCase):
    def test_a_gap_splits_the_wall(self):
        self.assertEqual(
            wall_with_gaps(300, 0, 300, 200, (60, 140)),
            [(300, 0, 300, 60), (300, 140, 300, 200)],
        )
        self.assertEqual(wall_with_gaps(0, 5, 10, 5), [(0, 5, 10, 5)])


class RoomTests(unittest.TestCase):
    def test_room_lookup_in_the_default_house(self):
        house = default_house()
        self.assertEqual(house.room_at(250, 200).name, "living_room")
        self.assertEqual(house.room_at(650, 100).name, "kitchen")
        self.assertEqual(house.room_at(100, 500).name, "office")
        self.assertEqual(house.room_at(650, 500).name, "bedroom")
        self.assertIsNone(house.room_at(-5, 5))

    def test_default_house_has_the_prd_things_in_their_rooms(self):
        house = default_house()
        by_room = {}
        for thing in house.things:
            self.assertEqual(house.room_at(thing.x, thing.y).name, thing.room, thing.name)
            by_room.setdefault(thing.room, set()).add(thing.name)
        self.assertEqual(
            by_room["living_room"], {"sofa", "television", "keyboard", "guitars", "backpack", "window"}
        )
        self.assertEqual(by_room["office"], {"desk", "monitor", "chair"})
        self.assertEqual(by_room["kitchen"], {"table", "coffee_machine", "window"})
        self.assertEqual(house.room_at(house.start.x, house.start.y).name, "living_room")

    def test_default_house_has_a_staircase(self):
        house = default_house()
        self.assertTrue(house.is_over_cliff(450, 350))
        self.assertFalse(house.is_over_cliff(300, 200))


class RayTests(unittest.TestCase):
    def test_distance_to_walls(self):
        house = two_rooms()
        self.assertAlmostEqual(house.ray_distance(50, 100, 90), 100)
        self.assertAlmostEqual(house.ray_distance(50, 100, 180), 50)
        self.assertAlmostEqual(house.ray_distance(50, 100, 270), 100)

    def test_the_doorway_is_open_and_the_range_is_capped(self):
        self.assertEqual(two_rooms().ray_distance(50, 100, 0), 400)
        self.assertAlmostEqual(two_rooms().ray_distance(50, 100, 0, max_range=1000), 550)

    def test_things_and_people_are_hit_before_the_wall(self):
        house = two_rooms()
        house.add_thing(Thing("box", 150, 100, 20))
        self.assertAlmostEqual(house.ray_distance(50, 100, 0), 80)
        house.add_person(Person("anna", 100, 100, 10))
        self.assertAlmostEqual(house.ray_distance(50, 100, 0), 40)

    def test_a_ray_that_misses_a_thing_passes(self):
        house = two_rooms()
        house.add_thing(Thing("box", 150, 160, 20))
        self.assertEqual(house.ray_distance(50, 100, 0), 400)

    def test_collision(self):
        house = two_rooms()
        house.add_thing(Thing("box", 150, 100, 20))
        self.assertTrue(house.collides(125, 100, 10))
        self.assertFalse(house.collides(100, 100, 10))
        self.assertTrue(house.collides(50, 195, 10))
        self.assertFalse(house.collides(300, 100, 10))


class VisibilityTests(unittest.TestCase):
    def test_field_of_view(self):
        house = two_rooms()
        house.add_thing(Thing("box", 150, 100, 10))
        self.assertEqual([t.name for t in house.visible(50, 100, 0)[0]], ["box"])
        self.assertEqual(house.visible(50, 100, 90)[0], [])
        self.assertEqual(house.visible(50, 100, 180)[0], [])

    def test_range_limit(self):
        house = two_rooms()
        house.add_thing(Thing("far", 550, 100, 10))
        self.assertEqual(house.visible(50, 100, 0, max_range=400)[0], [])
        self.assertEqual(len(house.visible(50, 100, 0, max_range=600)[0]), 1)

    def test_walls_block_the_view(self):
        house = two_rooms()
        house.add_thing(Thing("hidden", 450, 30, 10))
        house.add_thing(Thing("seen", 450, 100, 10))
        names = [t.name for t in house.visible(50, 100, 0)[0]]
        self.assertEqual(names, ["seen"])


class ChangeTests(unittest.TestCase):
    def test_things_move_and_change_room(self):
        house = two_rooms()
        house.add_thing(Thing("box", 150, 100, 20))
        self.assertEqual(house.thing("box").room, "a")
        house.move_thing("box", 450, 100)
        self.assertEqual(house.thing("box").room, "b")
        house.remove_thing("box")
        with self.assertRaises(KeyError):
            house.thing("box")

    def test_people_come_and_go(self):
        house = two_rooms()
        house.add_person(Person("anna", 100, 100))
        house.move_person("anna", 120, 100)
        self.assertEqual(house.people[0].x, 120)
        house.remove_person("anna")
        self.assertEqual(house.people, [])


if __name__ == "__main__":
    unittest.main()
