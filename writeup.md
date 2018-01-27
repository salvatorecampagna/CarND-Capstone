### Waypoint Updater

The `waypoint_updater` node is in charge of:

* defining a list of waypoints the car will follow
* setting the proper velocity for them.

The node listens for the `/base_waypoints, /current_pose` and `/traffic_waypoint` topics and publishes the new car trajectory 
on the `/final_waypoints` topic.

The node publishes a new trajecotry containing 50 waypoints at a rate of 50 Hz and, at each iteration, 
it checks first if a new red light is ahead of the car. If it is actually the case 
then the `waypoint_updater` plans a slow down until a full stop before the red light.

The knowledge of the presence of the red light comes from the tl detector node that 
publishes on the `/traffic_waypoint` topic the id of the waypoint of the red light.

Whenever a message is received from the `/traffic_waypoint`, the `traffic_cb` callback gets called and the id of the waypoint 
of the red light is saved.



 
