import React, { useEffect, useMemo, useRef, useState } from 'react'
import { NEOData } from '../utils/orbitalMath'

type Suggestion = {
  id: string
  name: string
  type?: string
}

export type SearchBoxProps = {
  neos: NEOData[]
  placeholder?: string
  maxSuggestions?: number
  onSelect: (neo: NEOData) => void
  onClear?: () => void
  autoFocus?: boolean
  compact?: boolean
}

// Utility: simple debounce
function useDebounced<T>(value: T, delayMs: number) {
  const [v, setV] = useState(value)
  useEffect(() => {
    const t = setTimeout(() => setV(value), delayMs)
    return () => clearTimeout(t)
  }, [value, delayMs])
  return v
}

export default function SearchBox({ neos, placeholder = 'Search NEO/PHA…', maxSuggestions = 15, onSelect, onClear, autoFocus = false, compact = false }: SearchBoxProps) {
  const [query, setQuery] = useState('')
  const debounced = useDebounced(query, 120)
  const [open, setOpen] = useState(false)
  const [activeIndex, setActiveIndex] = useState(0)
  const inputRef = useRef<HTMLInputElement | null>(null)
  const listRef = useRef<HTMLUListElement | null>(null)

  // Build a lightweight search index in-memory (lowercased)
  const index = useMemo(() => {
    return neos.map(n => ({ id: String(n.id), name: n.name, type: n.type, nameL: n.name.toLowerCase() }))
  }, [neos])

  // Compute suggestions locally (fast enough with debounce). Prefer prefix matches, then substrings.
  const suggestions: Suggestion[] = useMemo(() => {
    const q = debounced.trim().toLowerCase()
    if (!q) return []
    const prefix: Suggestion[] = []
    const contains: Suggestion[] = []
    for (let i = 0; i < index.length; i++) {
      const it = index[i]
      const pos = it.nameL.indexOf(q)
      if (pos === 0) {
        prefix.push({ id: it.id, name: it.name, type: it.type })
      } else if (pos > 0) {
        contains.push({ id: it.id, name: it.name, type: it.type })
      }
      if (prefix.length >= maxSuggestions) break
    }
    // Fill with contains if needed
    const needed = Math.max(0, maxSuggestions - prefix.length)
    return prefix.concat(contains.slice(0, needed))
  }, [debounced, index, maxSuggestions])

  useEffect(() => {
    setActiveIndex(0)
    setOpen(suggestions.length > 0)
  }, [suggestions.length])

  const handleKeyDown = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (!open || suggestions.length === 0) return
    if (e.key === 'ArrowDown') {
      e.preventDefault()
      setActiveIndex((i) => (i + 1) % suggestions.length)
    } else if (e.key === 'ArrowUp') {
      e.preventDefault()
      setActiveIndex((i) => (i - 1 + suggestions.length) % suggestions.length)
    } else if (e.key === 'Enter') {
      e.preventDefault()
      const s = suggestions[activeIndex]
      if (s) doSelect(s)
    } else if (e.key === 'Escape') {
      setOpen(false)
    }
  }

  const doSelect = (s: Suggestion) => {
    const neo = neos.find(n => String(n.id) === s.id)
    if (neo) {
      onSelect(neo)
      setQuery(neo.name)
      setOpen(false)
      setTimeout(() => inputRef.current?.blur(), 0)
    }
  }

  // Close when clicking outside
  useEffect(() => {
    const onDocClick = (ev: MouseEvent) => {
      if (!inputRef.current) return
      const target = ev.target as Node
      const container = inputRef.current.closest('.neo-search')
      if (container && target && container.contains(target)) return
      setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [])

  return (
    <div className={`neo-search ${compact ? 'neo-search--compact' : ''}`} onMouseDown={(e) => e.stopPropagation()}>
      <div className="neo-search__inputWrap">
        <span className="neo-search__icon" aria-hidden>
          <i className="fa-solid fa-magnifying-glass" />
        </span>
        <input
          ref={inputRef}
          className="neo-search__input"
          type="text"
          role="combobox"
          aria-expanded={open}
          aria-controls="neo-search-listbox"
          aria-autocomplete="list"
          placeholder={placeholder}
          value={query}
          onChange={e => setQuery(e.target.value)}
          onFocus={() => setOpen(suggestions.length > 0)}
          onKeyDown={handleKeyDown}
          autoFocus={autoFocus}
        />
        {query && (
          <button className="neo-search__clear" onMouseDown={(e) => { e.preventDefault(); setQuery(''); setOpen(false); onClear?.(); }} aria-label="Clear">
            <i className="fa-solid fa-xmark" />
          </button>
        )}
      </div>
      {open && suggestions.length > 0 && (
        <ul id="neo-search-listbox" role="listbox" ref={listRef} className="neo-search__list">
          {suggestions.map((s, idx) => (
            <li
              key={`${s.id}-${idx}`}
              role="option"
              aria-selected={idx === activeIndex}
              className={`neo-search__item ${idx === activeIndex ? 'is-active' : ''}`}
              onMouseEnter={() => setActiveIndex(idx)}
              onMouseDown={(e) => { e.preventDefault(); doSelect(s) }}
            >
              <span className="neo-search__name">{s.name}</span>
              {s.type && <span className={`neo-search__tag ${s.type === 'PHA' ? 'is-pha' : 'is-neo'}`}>{s.type}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
