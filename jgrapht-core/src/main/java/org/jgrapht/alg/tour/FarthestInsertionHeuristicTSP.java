package org.jgrapht.alg.tour;

import org.jgrapht.Graph;
import org.jgrapht.GraphPath;
import org.jgrapht.alg.tour.HamiltonianCycleAlgorithmBase;

import java.util.*;
import java.util.stream.Collectors;

/**
 * The farthest insertion heuristic algorithm for the TSP.
 * From a starting subtour, the vertex v, whose distance to the tour is maximal, is chosen to be inserted.
 * The insertion is done either to the 'left' or the 'right' of the tour vertex, which is closest to v.
 * Core from org.jgrapht.alg.tour.NearestInsertionHeuristicTSP, deviations made to solve the farthest insertion heuristic.
 *
 * @author Peter Harman, Carl-Elias Goldberg
 */
public class FarthestInsertionHeuristicTSP<V, E>
        extends
        HamiltonianCycleAlgorithmBase<V, E> {

    private final GraphPath<V, E> subtour;

    /**
     * Constructor. By default, a sub-tour is chosen based on the longest edge
     */
    public FarthestInsertionHeuristicTSP() {
        this(null);
    }

    /**
     * Constructor
     * Specifies an existing sub-tour that will be augmented to form a complete tour when
     * {@link #getTour(org.jgrapht.Graph) } is called
     *
     * @param subtour Initial sub-tour, or null to start with the longest edge
     */
    public FarthestInsertionHeuristicTSP(GraphPath<V, E> subtour) {
        this.subtour = subtour;
    }

    /**
     * Computes a tour using the farthest insertion heuristic.
     *
     * @param graph the input graph
     * @return a tour
     * @throws IllegalArgumentException If the graph is not undirected
     * @throws IllegalArgumentException If the graph is not complete
     * @throws IllegalArgumentException If the graph contains no vertices
     * @throws IllegalArgumentException If the specified sub-tour is for a different Graph instance
     * @throws IllegalArgumentException If the graph does not contain specified sub-tour vertices
     * @throws IllegalArgumentException If the graph does not contain specified sub-tour edges
     */
    @Override
    public GraphPath<V, E> getTour(Graph<V, E> graph) {
        checkGraph(graph);
        if (graph.vertexSet().size() == 1) {
            return getSingletonTour(graph);
        }
        return vertexListToTour(augment(subtour(graph), graph), graph);
    }

    /**
     * Get or create a sub-tour to start augmenting
     *
     * @param graph The graph
     * @return Vertices of an initial sub-tour
     * @throws IllegalArgumentException If the specified sub-tour is for a different Graph instance
     * @throws IllegalArgumentException If the graph does not contain specified sub-tour vertices
     * @throws IllegalArgumentException If the graph does not contain specified sub-tour edges
     */
    private List<V> subtour(Graph<V, E> graph) {
        List<V> subtourVertices = new ArrayList<>();
        if (subtour != null) {
            if (subtour.getGraph() != null && !graph.equals(subtour.getGraph())) {
                throw new IllegalArgumentException(
                        "Specified sub-tour is for a different Graph instance");
            }
            if (!graph.vertexSet().containsAll(subtour.getVertexList())) {
                throw new IllegalArgumentException(
                        "Graph does not contain specified sub-tour vertices");
            }
            if (!graph.edgeSet().containsAll(subtour.getEdgeList())) {
                throw new IllegalArgumentException(
                        "Graph does not contain specified sub-tour edges");
            }
            if (subtour.getStartVertex().equals(subtour.getEndVertex())) {
                subtourVertices
                        .addAll(subtour.getVertexList().subList(1, subtour.getVertexList().size()));
            } else {
                subtourVertices.addAll(subtour.getVertexList());
            }
        }
        if (subtourVertices.isEmpty()) {
            // If no initial subtour exists, create one based on the longest edge
            E longestEdge = Collections
                    .max(
                            graph.edgeSet(),
                            Comparator.comparingDouble(graph::getEdgeWeight));
            subtourVertices.add(graph.getEdgeSource(longestEdge));
            subtourVertices.add(graph.getEdgeTarget(longestEdge));
        }
        return subtourVertices;
    }

    /**
     * Initialise the Map storing the closest tour vertex for each non-tour vertex
     *
     * @param tourVertices Current tour vertices
     * @param unvisited    Set of unvisited vertices (non-tour vertices)
     * @param graph        The graph
     * @return Map storing the closest tour vertex for each non-tour vertex
     */
    private Map<V, FarthestInsertionHeuristicTSP.Closest<V>> getClosest(List<V> tourVertices, Set<V> unvisited, Graph<V, E> graph) {
        return unvisited
                .stream().collect(Collectors.toMap(v -> v, v -> getClosest(v, tourVertices, graph)));
    }

    /**
     * Determines closest tour-vertex to a vertex not in the current tour
     *
     * @param nonTourVertex Vertex not in the current tour
     * @param tourVertices  List of vertices in the current tour
     * @param graph         The graph
     * @return Closest tour vertex
     */
    private FarthestInsertionHeuristicTSP.Closest<V> getClosest(V nonTourVertex, List<V> tourVertices, Graph<V, E> graph) {
        V closest = null;
        double minDist = Double.MAX_VALUE;
        for (V tourVertex : tourVertices) {
            double vDist = graph.getEdgeWeight(graph.getEdge(nonTourVertex, tourVertex));
            if (vDist < minDist) {
                closest = tourVertex;
                minDist = vDist;
            }
        }
        return new FarthestInsertionHeuristicTSP.Closest<>(closest, nonTourVertex, minDist);
    }

    /**
     * Update the Map storing the closest tour vertex for each non-tour vertex
     *
     * @param currentClosest Map storing the closest tour vertex for each non-tour vertex
     * @param chosen         Latest vertex added to tour
     * @param unvisited      Set of unvisited vertices
     */
    private void updateMap(
            Map<V, FarthestInsertionHeuristicTSP.Closest<V>> currentClosest, FarthestInsertionHeuristicTSP.Closest<V> chosen, Set<V> unvisited, Graph<V, E> graph) {
        V newTourVertex = chosen.unvisitedVertex();
        // Update the set of unvisited vertices, and exit if none remain
        unvisited.remove(newTourVertex);
        if (unvisited.isEmpty()) {
            currentClosest.clear();
            return;
        }

        currentClosest.remove(newTourVertex);
        currentClosest.replaceAll((v, c) -> {
            double distToNewTourVertex = graph.getEdgeWeight(graph.getEdge(newTourVertex, c.unvisitedVertex()));
            if (distToNewTourVertex < c.distance()) {
                return (new Closest<>(newTourVertex, c.unvisitedVertex(), distToNewTourVertex));
            } else return c;
        });
    }

    /**
     * Chooses the unvisited vertex, which is farthest to the sub-tour (its minimum distance to the tour is maximal)
     *
     * @param closestVertices Map storing the closest tour vertex for each tour vertex
     * @return First result of sorting values
     */
    private FarthestInsertionHeuristicTSP.Closest<V> chooseFarthest(Map<V, FarthestInsertionHeuristicTSP.Closest<V>> closestVertices) {
        return Collections.max(closestVertices.values());
    }

    /**
     * Augment an existing tour to give a complete tour
     *
     * @param subtour The vertices of the existing tour
     * @param graph   The graph
     * @return List of vertices representing the complete tour
     */
    private List<V> augment(List<V> subtour, Graph<V, E> graph) {
        Set<V> unvisited = new HashSet<>(graph.vertexSet());
        subtour.forEach(unvisited::remove);
        return augment(subtour, getClosest(subtour, unvisited, graph), unvisited, graph);
    }

    /**
     * Augment an existing tour to give a complete tour
     *
     * @param subtour         The vertices of the existing tour
     * @param closestVertices Map of data for farthest unvisited vertices
     * @param unvisited       Set of unvisited vertices
     * @param graph           The graph
     * @return List of vertices representing the complete tour
     */
    private List<V> augment(
            List<V> subtour, Map<V, FarthestInsertionHeuristicTSP.Closest<V>> closestVertices, Set<V> unvisited, Graph<V, E> graph) {
        while (!unvisited.isEmpty()) {
            // Select a city not in the subtour, having the farthest distance to the closest city in the subtour.
            FarthestInsertionHeuristicTSP.Closest<V> farthestVertex = chooseFarthest(closestVertices);

            // Determine the vertices either side of the selected tour vertex
            int i = subtour.indexOf(farthestVertex.tourVertex());
            V vertexBefore = subtour.get(i == 0 ? subtour.size() - 1 : i - 1);
            V vertexAfter = subtour.get(i == subtour.size() - 1 ? 0 : i + 1);

            // Find an edge in the subtour such that the cost of inserting the selected city between
            // the edge’s cities will be minimal.
            // Making assumption this is a neighbouring edge, test the edges before and after
            double insertionCostBefore =
                    graph.getEdgeWeight(graph.getEdge(vertexBefore, farthestVertex.unvisitedVertex()))
                            + graph.getEdgeWeight(graph.getEdge(farthestVertex.tourVertex(), farthestVertex.unvisitedVertex()))
                            - graph.getEdgeWeight(graph.getEdge(vertexBefore, farthestVertex.tourVertex()));
            double insertionCostAfter =
                    graph.getEdgeWeight(graph.getEdge(vertexAfter, farthestVertex.unvisitedVertex()))
                            + graph.getEdgeWeight(graph.getEdge(farthestVertex.tourVertex(), farthestVertex.unvisitedVertex()))
                            - graph.getEdgeWeight(graph.getEdge(vertexAfter, farthestVertex.tourVertex()));

            // Add the selected vertex to the tour
            if (insertionCostBefore < insertionCostAfter) {
                subtour.add(i, farthestVertex.unvisitedVertex());
            } else {
                subtour.add(i + 1, farthestVertex.unvisitedVertex());
            }

            // Repeat until no more cities remain
            updateMap(closestVertices, farthestVertex, unvisited, graph);
        }
        return subtour;
    }

    /**
     * Class holding data for the closest unvisited vertex to a particular vertex in the tour.
     *
     * @param <V> vertex type
     */
    private record Closest<V>(V tourVertex, V unvisitedVertex, double distance)
            implements
            Comparable<Closest<V>> {

        @Override
        public int compareTo(FarthestInsertionHeuristicTSP.Closest<V> o) {
            return Double.compare(distance, o.distance);
        }

    }
}

